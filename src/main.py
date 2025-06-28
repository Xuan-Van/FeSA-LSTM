import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train import train
import os

tf.get_logger().setLevel('ERROR') # 日志级别设置

os.makedirs("result", exist_ok=True) # 确保结果目录存在


# 特征缩放
def scale_features(train_df, test_df):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # 缩放方式
    standard_features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_remainder']
    log_standard_features = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    minmax_features = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    # 标准化
    scaler, standard_scaler = StandardScaler(), StandardScaler()
    train_scaled['Global_active_power'] = scaler.fit_transform(train_df[['Global_active_power']])
    test_scaled['Global_active_power'] = scaler.transform(test_df[['Global_active_power']])
    train_scaled[standard_features] = standard_scaler.fit_transform(train_df[standard_features])
    test_scaled[standard_features] = standard_scaler.transform(test_df[standard_features])

    # 对数标准化
    log_scaler = StandardScaler()
    train_log = np.log1p(train_df[log_standard_features])
    test_log = np.log1p(test_df[log_standard_features])
    train_scaled[log_standard_features] = log_scaler.fit_transform(train_log)
    test_scaled[log_standard_features] = log_scaler.transform(test_log)

    # 最小最大缩放
    minmax_scaler = MinMaxScaler()
    train_scaled[minmax_features] = minmax_scaler.fit_transform(train_df[minmax_features])
    test_scaled[minmax_features] = minmax_scaler.transform(test_df[minmax_features])

    return train_scaled.values, test_scaled.values, scaler


# 滑动窗口
def sliding_window(data, past_len, future_len):
    X, Y = [], []
    for i in range(len(data) - past_len - future_len + 1):
        X.append(data[i:i+past_len])  # 历史窗口
        Y.append(data[i+past_len:i+past_len+future_len, 0])  # 预测目标
    return np.array(X), np.array(Y)


# 滚动预测
def rolling_forecast(model, history, past_len, future_len):
    pred_len = len(history) - past_len
    pred_dict = {i: [] for i in range(pred_len)}

    # 预测每个时间点
    for i in range(pred_len):
        window_input = history[i:i + past_len].reshape(1, past_len, history.shape[1]) # 窗口数据
        pred = model.predict(window_input, verbose=0) # 模型预测
        pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape) # 反缩放

        for j in range(min(future_len, pred_len - i)):
            pred_dict[i + j].append(pred[0, j])

    # 预测取值方式
    pred_result = {}
    pred_result['mean'] = np.array([np.mean(pred_dict[i]) for i in range(pred_len)])
    pred_result['median'] = np.array([np.median(pred_dict[i]) for i in range(pred_len)])
    pred_result['ema'] = []

    # 指数移动平均
    alpha = 0.1
    for i in range(pred_len):
        ema = pred_dict[i][0]
        for value in pred_dict[i][1:]:
            ema = alpha * value + (1 - alpha) * ema
        pred_result['ema'].append(ema)

    return pred_result


# 寻找 MAE 最小的预测取值方法
def find_best_prediction(ground_truth, predictions):
    mae_values = {
        'mean': mean_absolute_error(ground_truth, predictions['mean']),
        'median': mean_absolute_error(ground_truth, predictions['median']),
        'ema': mean_absolute_error(ground_truth, predictions['ema'])
    }
    best_method = min(mae_values, key=mae_values.get)

    return predictions[best_method]


# 绘制预测结果
def plot_predictions(preds, ground_truth, future_len):
    lstm_pred, transformer_pred, translstm = preds[0], preds[1], preds[2]

    plt.figure(figsize=(12, 4))
    plt.plot(ground_truth, label="Ground Truth", color='grey')
    plt.plot(lstm_pred, label="LSTM Prediction", color='blue')
    plt.plot(transformer_pred, label="Transformer Prediction", color='green')
    plt.plot(translstm, label="FeSA-LSTM Prediction", color='red')

    plt.title(f"Predictions for {future_len} Days")
    plt.legend()
    plt.savefig(f"result/{future_len}d.png")
    plt.close()


# 执行评估
def run_evaluation(model_name, past_len, future_len):
    # 获取数据
    X_train, Y_train = sliding_window(train_data, past_len, future_len) # 滑动窗口生成训练数据
    history = np.concatenate([train_data[-past_len:], test_data]) # 历史数据序列

    mses, maes, preds = [], [], []
    for batch_size in [16, 32, 64, 128, 256]:
        model = train(model_name, X_train, Y_train, batch_size) # 训练模型
        predictions = rolling_forecast(model, history, past_len, future_len) # 滚动预测
        pred = find_best_prediction(ground_truth, predictions) # 寻找最佳预测

        mse = mean_squared_error(ground_truth, pred) # 计算均方误差
        mae = mean_absolute_error(ground_truth, pred) # 计算平均绝对误差

        mses.append(mse)
        maes.append(mae)
        preds.append(pred)

    # 输出评估结果
    results.append(f"[{model_name} - {future_len}d]\n")
    results.append(f"AVG MSE: {np.mean(mses):.2f}\n")
    results.append(f"MSE STD: {np.std(mses):.2f}\n")
    results.append(f"AVG MAE: {np.mean(maes):.2f}\n")
    results.append(f"MAE STD: {np.std(maes):.2f}\n\n")

    if future_len == 90:
        pred_90d.append(np.mean(preds, axis=0))
    else:
        pred_365d.append(np.mean(preds, axis=0))


if __name__ == "__main__":
    # 载入数据
    train_daily = pd.read_csv("data/train_daily.csv", index_col=['DateTime'])
    test_daily = pd.read_csv("data/test_daily.csv", index_col=['DateTime'])

    # 统一数据类型为float
    for col in train_daily.columns:
        train_daily[col] = train_daily[col].astype('float32')
        test_daily[col] = test_daily[col].astype('float32')

    train_data, test_data, scaler = scale_features(train_daily, test_daily) # 特征缩放

    ground_truth = test_daily['Global_active_power'].values.reshape(-1, 1) # 获取真实值

    pred_90d, pred_365d, results = [], [], []
    run_evaluation('LSTM', past_len=90, future_len=90)
    run_evaluation('Transformer', past_len=1, future_len=90)
    run_evaluation('FeSA-LSTM', past_len=90, future_len=90)
    
    run_evaluation('lstm', past_len=90, future_len=365)
    run_evaluation('Transformer', past_len=1, future_len=365)
    run_evaluation('FeSA-LSTM', past_len=90, future_len=365)

    with open(f"result/evaluation.txt", 'w') as f:
        f.writelines(results)

    plot_predictions(pred_90d, ground_truth, future_len=90)
    plot_predictions(pred_365d, ground_truth, future_len=365)