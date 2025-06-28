import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

os.makedirs("figure", exist_ok=True) # 确保目录存在


# 载入并清洗数据
def load_and_clean(filepath):
    column_names = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    df = pd.read_csv(
        filepath,
        names=column_names,
        header=0 if 'train' in filepath else None, # 测试集没有表头，需要指定
        low_memory=False, # 处理大文件时避免内存问题
        index_col=['DateTime'] # 将 DateTime 列设置为索引
    )
    df.index = pd.to_datetime(df.index) # 转换索引为日期时间格式
    df.replace('?', np.nan, inplace=True) # 替换缺失值标记

    # 填充缺失值
    df = df.fillna(df.shift(1440)) # 用前一天的数据填充
    df = df.fillna(df.shift(2880)) # 用两天前的数据填充
    df = df.fillna(df.shift(-1440)) # 用后一天的数据填充
    df = df.fillna(df.shift(-2880)) # 用两天后数据填充

    # 统一数据类型为float
    for col in df.columns:
        df[col] = df[col].astype('float32')

    df['RR'] = df['RR'] / 10 # 单位变换

    # 计算剩余电量
    sub_metering_remainder = (df['Global_active_power'] * 1000 / 60) - (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    index_of_sub_metering_3 = df.columns.get_loc('Sub_metering_3')
    df.insert(index_of_sub_metering_3 + 1, 'Sub_metering_remainder', sub_metering_remainder)

    return df


# 绘制特征子图
def plot_all_features(df, plot_type='line'):
    num_features = df.shape[1]
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 3 * num_features), constrained_layout=True)

    for i, col in enumerate(df.columns):
        ax = axs[i]
        if plot_type == 'line':
            ax.plot(df.index, df[col], label=col, linewidth=0.8)
        elif plot_type == 'hist':
            ax.hist(df[col].dropna(), bins=50, label=col, color='skyblue', edgecolor='black')
        ax.set_title(f"Feature: {col}")
        ax.legend()

    plt.suptitle(f"All Features ({plot_type})")
    plt.savefig(f"figure/all_features_{plot_type}.png")
    plt.close()


# 绘制年度有功功率子图
def plot_gap_by_year(df, plot_type='line'):
    df = df.copy()
    df['Year'] = df.index.year
    years = sorted(df['Year'].unique())

    fig, axs = plt.subplots(len(years), 1, figsize=(12, 3 * len(years)), constrained_layout=True)

    for i, year in enumerate(years):
        ax = axs[i]
        yearly_data = df[df['Year'] == year]['Global_active_power']
        if plot_type == 'line':
            ax.plot(yearly_data.index, yearly_data, label=str(year))
        elif plot_type == 'hist':
            ax.hist(yearly_data.dropna(), bins=50, label=str(year), color='lightcoral', edgecolor='black')
        ax.set_title(f"Global Active Power - {year}")
        ax.legend()

    plt.suptitle(f"Global Active Power by Year ({plot_type})")
    plt.savefig(f"figure/gap_by_year_{plot_type}.png")
    plt.close()


# 绘制月度有功功率子图
def plot_gap_by_month(year, df, plot_type='line'):
    df = df[df.index.year == year]
    fig, axs = plt.subplots(12, 1, figsize=(12, 36), constrained_layout=True)

    for month in range(1, 13):
        ax = axs[month - 1]
        monthly_data = df[df.index.month == month]['Global_active_power']
        if plot_type == 'line':
            ax.plot(monthly_data.index, monthly_data, label=f"{year}-{month:02d}")
        elif plot_type == 'hist':
            ax.hist(monthly_data.dropna(), bins=50, label=f"{year}-{month:02d}", color='orange', edgecolor='black')
        ax.set_title(f"Global Active Power - {year}-{month:02d}")
        ax.legend()

    plt.suptitle(f"Global Active Power by Month in {year} ({plot_type})")
    plt.savefig(f"figure/gap_by_month_{year}_{plot_type}.png")
    plt.close()


# 绘制日度有功功率子图
def plot_gap_by_day(year, month, df, plot_type='line'):
    df = df[(df.index.year == year) & (df.index.month == month)]
    days = sorted(df.index.day.unique())

    fig, axs = plt.subplots(len(days), 1, figsize=(12, 3 * len(days)), constrained_layout=True)

    for i, day in enumerate(days):
        ax = axs[i]
        daily_data = df[df.index.day == day]['Global_active_power']
        if plot_type == 'line':
            ax.plot(daily_data.index, daily_data, label=f"{year}-{month:02d}-{day:02d}")
        elif plot_type == 'hist':
            ax.hist(daily_data.dropna(), bins=50, label=f"{year}-{month:02d}-{day:02d}", color='green', edgecolor='black')
        ax.set_title(f"Global Active Power - {year}-{month:02d}-{day:02d}")
        ax.legend()

    plt.suptitle(f"Global Active Power by Day: {year}-{month:02d} ({plot_type})")
    plt.savefig(f"figure/gap_by_day_{year}_{month:02d}_{plot_type}.png")
    plt.close()


# 数据按日聚合
def aggregate_daily(df):
    daily = pd.DataFrame()

    # 取总和
    for col in ["Global_active_power", "Global_reactive_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder"]:
        daily[col] = df[col].resample("D").sum()

    # 取平均值
    for col in ["Voltage", "Global_intensity"]:
        daily[col] = df[col].resample("D").mean()

    # 取当日第一个值
    for col in ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        daily[col] = df[col].resample("D").first()

    return daily


if __name__ == "__main__":
    # 载入数据并清洗
    train_df = load_and_clean("data/train.csv")
    test_df = load_and_clean("data/test.csv")
    all_df = pd.concat([train_df, test_df])
    
    # 所有特征
    plot_all_features(all_df, plot_type='line')
    plot_all_features(all_df, plot_type='hist')

    # 每年
    plot_gap_by_year(all_df, plot_type='line')
    plot_gap_by_year(all_df, plot_type='hist')

    # 指定某年
    plot_gap_by_month(2008, all_df, plot_type='line')
    plot_gap_by_month(2008, all_df, plot_type='hist')

    # 指定某年某月
    plot_gap_by_day(2008, 8, all_df, plot_type='line')
    plot_gap_by_day(2008, 8, all_df, plot_type='hist')

    # 按日聚合
    train_daily = aggregate_daily(train_df)
    test_daily = aggregate_daily(test_df)

    # 保存数据
    train_daily.to_csv("data/train_daily.csv")
    test_daily.to_csv("data/test_daily.csv")