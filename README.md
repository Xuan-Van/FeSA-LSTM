## 虚拟环境

```bash
conda create -n time python==3.7.6 -y
conda activate time
pip install tensorflow==2.10.0 pandas matplotlib scikit-learn 
```

## 项目结构

```
data/
    test.csv
    test_daily.csv
    train.csv
    train_daily.csv

figure/
    all_features_hist.png
    all_features_line.png
    gap_by_day_2008_08_hist.png
    gap_by_day_2008_08_line.png
    gap_by_month_2008_hist.png
    gap_by_month_2008_line.png
    gap_by_year_hist.png
    gap_by_year_line.png

result/
    365d.png
    90d.png
    evaluation.txt

src/
    main.py
    preprocess.py
    train.py
```

## 脚本运行

```bash
python src/preprocess.py
python src/main.py
```

## 数据集来源

1. 家庭用电数据：[Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. 天气数据：[Basic climatological data - monthly](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)