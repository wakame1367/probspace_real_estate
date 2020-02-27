from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nyaggle.experiment import run_experiment
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

data_path = Path("resources")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def load_dataset():
    train = pd.read_csv(data_path / "train_data.csv")
    test = pd.read_csv(data_path / "test_data.csv")
    return train, test


def category_encode(train, test, target_cols):
    _all = pd.concat([train, test])
    for col in target_cols:
        train[col] = train[col].fillna("NaN")
        test[col] = test[col].fillna("NaN")
        _all[col] = _all[col].fillna("NaN")
        le = LabelEncoder()
        le.fit(_all[col])
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    return train, test


def built_year(df):
    df['建築年'] = df['建築年'].dropna()
    df['建築年'] = df['建築年'].str.replace('戦前', '昭和20年')
    df['年号'] = df['建築年'].str[:2]
    df['和暦年数'] = df['建築年'].str[2:].str.strip('年').fillna(
        0).astype(int)
    df.loc[df['年号'] == '昭和', '建築年'] = df['和暦年数'] + 1925
    df.loc[df['年号'] == '平成', '建築年'] = df['和暦年数'] + 1988
    return df


def walk_time(df):
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace('30分?60分', '45')
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace('1H?1H30', '75')
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace('1H30?2H', '105')
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace('2H?', '120')
    df['最寄駅：距離（分）'] = pd.to_numeric(df['最寄駅：距離（分）'],
                                    errors='coerce')
    return df


def area1(df):
    replace_dict = {'10m^2未満': 9, '2000㎡以上': 2000}
    df['延床面積（㎡）'] = pd.to_numeric(df['延床面積（㎡）'].replace(replace_dict))
    return df


def area2(df):
    replace_dict = {'2000㎡以上': 2000, '5000㎡以上': 5000}
    df['面積（㎡）'] = pd.to_numeric(df['面積（㎡）'].replace(replace_dict))
    return df


def maguchi(df):
    df['間口'] = pd.to_numeric(df['間口'].replace('50.0m以上', 50.0))
    return df


def preprocess(df):
    df = built_year(df)
    df = walk_time(df)
    df = area1(df)
    df = area2(df)
    df = maguchi(df)
    return df


def main():
    with open("settings/colum_names.yml", "r", encoding="utf-8") as f:
        rename_dict = yaml.load(f, Loader=yaml.Loader)

    submit = pd.read_csv("resources/submission.csv")
    train, test = load_dataset()
    target_col = "y"
    target = train[target_col]
    target = target.map(np.log1p)
    train.drop(columns=[target_col], inplace=True)
    train = preprocess(train)
    test = preprocess(test)
    drop_cols = ["id", "都道府県名", "市区町村名", "年号", "和暦年数", "建築年"]
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)

    train = train.rename(columns=rename_dict)
    test = test.rename(columns=rename_dict)
    cat_cols = ['Type', 'Region', 'MunicipalityCode', 'DistrictName',
                'NearestStation', 'FloorPlan', 'LandShape', 'Structure', 'Use',
                'Purpose', 'Direction', 'Classification', 'CityPlanning',
                'Renovation', 'Remarks', 'Period']
    train, test = category_encode(train, test, cat_cols)

    lightgbm_params = {
        'max_depth': 8
    }

    kf = KFold(n_splits=4)

    run_experiment(lightgbm_params,
                   X_train=train,
                   y=target,
                   X_test=test,
                   eval_func=rmse,
                   cv=kf,
                   logging_directory='resources/logs/{time}',
                   sample_submission=submit)


if __name__ == '__main__':
    main()
