import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam


# 构建 LSTM 模型
def build_lstm_model(input_shape, output_len):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs) # 第一层双向 LSTM
    x = layers.Bidirectional(layers.LSTM(64))(x) # 第二层双向 LSTM
    x = layers.Dense(output_len)(x) # 全连接输出层
    return Model(inputs, x)


# 构建 Transformer 模型
def build_transformer_model(input_shape, output_len, num_heads=8, d_model=256):
    inputs = layers.Input(shape=input_shape)

    # 位置编码
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)

    x = layers.Dense(d_model)(inputs) # 全连接输入层
    x += position_embedding # 添加位置编码

    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x) # 注意力层
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output) # 残差连接和层归一化

    ffn = layers.Dense(d_model*2, activation="gelu")(x) # GELU 激活函数
    ffn = layers.Dense(d_model)(ffn) # 前馈神经网络
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn) # 残差连接和层归一化

    x = layers.GlobalAveragePooling1D()(x) # 全局平均池化
    x = layers.Dense(output_len)(x) # 全连接输出层
    return Model(inputs, x)


# 构建 FeSA-LSTM 模型
def build_fesalstm_model(input_shape, output_len, d_model=260):
    inputs = layers.Input(shape=input_shape)
    num_heads = input_shape[1]

    # 位置编码
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)

    x = layers.Dense(d_model)(inputs) # 全连接输入层
    x += position_embedding # 添加位置编码

    # 分割特征：每个特征一个通道
    split_size = d_model // num_heads # 每个特征的维度
    feature_splits = tf.split(x, num_or_size_splits=[split_size] * num_heads, axis=-1)

    # 为每个特征创建独立的注意力头
    attn_outputs = []
    for i, feature in enumerate(feature_splits):
        attn_layer = layers.MultiHeadAttention(num_heads=1, key_dim=split_size) # 单头注意力
        attn_out = attn_layer(feature, feature) # 自注意力机制
        attn_outputs.append(attn_out)

    x = tf.concat(attn_outputs, axis=-1) # 合并所有注意力头的输出
    residual_projection = layers.Dense(d_model)(inputs) # 残差投影
    x = layers.LayerNormalization(epsilon=1e-6)(residual_projection + x)  # 残差连接和层归一化

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x) # 第一层双向 LSTM
    x = layers.Bidirectional(layers.LSTM(64))(x) # 第二层双向 LSTM
    x = layers.Dense(output_len)(x) # 全连接输出层
    return Model(inputs, x)


# 训练
def train(model_name, X_train, Y_train, batch_size):
    if model_name == "LSTM":
        model = build_lstm_model(X_train.shape[1:], Y_train.shape[1])
    elif model_name == "Transformer":
        model = build_transformer_model(X_train.shape[1:], Y_train.shape[1])
    else:
        model = build_fesalstm_model(X_train.shape[1:], Y_train.shape[1])

    optimizer = Adam(learning_rate=0.001) # 创建优化器
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=batch_size,
        verbose=2
    )
    return model