from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nyaggle.experiment import run_experiment
from nyaggle.hyper_parameters.lightgbm import parameters
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

data_path = Path("resources")

parameters


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
    df['和暦年数'] = df['建築年'].str[2:].str.strip('年').fillna(0).astype(int)
    df.loc[df['年号'] == '昭和', '建築年'] = df['和暦年数'] + 1925
    df.loc[df['年号'] == '平成', '建築年'] = df['和暦年数'] + 1988
    df['建築年'] = pd.to_numeric(df['建築年'])
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


def preprocess_land_price(train, test):
    land_price = pd.read_csv("resources/published_land_price.csv")
    # land_price = land_price.rename({"緯度": "latitude", "経度": "longitude"})
    land_price = clean_land_price(land_price)
    train = clean_train_test(train)
    test = clean_train_test(test)
    train, test = add_landp(train, test, land_price)
    return train, test


def clean_land_price(df):
    target_col = "市区町村名"
    # 東京府中 -> 府中
    df[target_col] = df[target_col].replace(r"^東京", "", regex=True)
    return df


def clean_train_test(df):
    target_col = "市区町村名"
    # 西多摩郡日の出 -> 日の出
    df[target_col] = df[target_col].replace(r"^西多摩郡", "", regex=True)
    df[target_col] = df[target_col].map(lambda x: x.rstrip("市区町村"))
    return df


def add_landp(train, test, land_price):
    # 直近5年のみ対象
    target_cols = ["Ｈ２７価格", "Ｈ２８価格", "Ｈ２９価格", "Ｈ３０価格", "Ｈ３１価格"]
    land_price["landp_mean"] = land_price[target_cols].mean(axis=1)
    landp_mean = land_price.groupby("市区町村名")["landp_mean"].mean().reset_index()
    train = train.merge(landp_mean, on='市区町村名')
    test = test.merge(landp_mean, on='市区町村名')
    return train, test


def add_lat_and_long(train, test, land_price):
    lat_and_long = land_price.groupby("市区町村名")[
        "latitude", "longitude"].mean().reset_index()
    train = train.merge(lat_and_long, on='市区町村名')
    test = test.merge(lat_and_long, on='市区町村名')
    return train, test


def make_sample_submission(test, target_col):
    test_copy = test.copy()
    index_col = "id"
    submit_path = Path("resources/submission.csv")
    test_copy.loc[:, target_col] = 0
    submit = test_copy[[index_col, target_col]]
    if not submit_path.exists():
        submit.to_csv(submit_path, index=False)
    return submit


def main():
    with open("settings/colum_names.yml", "r", encoding="utf-8") as f:
        rename_dict = yaml.load(f, Loader=yaml.Loader)

    train, test = load_dataset()

    target_col = "y"
    submit = make_sample_submission(test, target_col)
    target = train[target_col]
    target = target.map(np.log1p)
    train.drop(columns=[target_col], inplace=True)
    train = preprocess(train)
    test = preprocess(test)
    drop_cols = ["id", "都道府県名", "市区町村名", "年号", "和暦年数"]
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
        "metric": "rmse",
        "objective": 'regression',
        "max_depth": 5,
        "num_leaves": 24,
        "learning_rate": 0.007,
        "n_estimators": 30000,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 1,
        "reg_alpha": 0,
        "reg_lambda": 0,
    }

    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 5000
    }

    kf = KFold(n_splits=4)

    lgb_result = run_experiment(lightgbm_params,
                                X_train=train,
                                y=target,
                                X_test=test,
                                eval_func=rmse,
                                cv=kf,
                                fit_params=fit_params,
                                logging_directory='resources/logs/'
                                                  'lightgbm/{time}',
                                sample_submission=submit)

    catboost_params = {
        'learning_rate': 0.01,
        'max_depth': 8,
        'bagging_temperature': 0.8,
        'l2_leaf_reg': 45,
        'od_type': 'Iter'
    }

    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 5000
    }

    # cab_result = run_experiment(catboost_params,
    #                             X_train=train,
    #                             y=target,
    #                             X_test=test,
    #                             eval_func=rmse,
    #                             cv=kf,
    #                             categorical_feature=cat_cols,
    #                             fit_params=fit_params,
    #                             algorithm_type='cat',
    #                             logging_directory='resources/logs/'
    #                                               'catboost/{time}',
    #                             sample_submission=submit)


if __name__ == '__main__':
    main()
