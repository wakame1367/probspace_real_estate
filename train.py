from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nyaggle.experiment import run_experiment
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from preprocess import preprocess, category_encode

data_path = Path("resources")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def load_dataset():
    train = pd.read_csv(data_path / "train_data.csv")
    test = pd.read_csv(data_path / "test_data.csv")
    return train, test


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
    is_train = "is_train"
    submit = make_sample_submission(test, target_col)
    target = train[target_col]
    target = target.map(np.log1p)
    train[is_train] = 1
    test[is_train] = 0
    train.drop(columns=[target_col], inplace=True)
    _all = pd.concat([train, test], ignore_index=True)
    _all = _all.rename(columns=rename_dict)
    _all = preprocess(_all)
    drop_cols = ["id", "Prefecture", "Municipality", "年号", "和暦年数"]
    one_hot_cols = ['Structure', 'Use', 'Remarks']
    cat_cols = ['Type', 'Region', 'MunicipalityCode', 'DistrictName',
                'NearestStation', 'FloorPlan', 'LandShape', 'Purpose',
                'Direction', 'Classification', 'CityPlanning', 'Renovation',
                'Period']
    _all.drop(columns=drop_cols, inplace=True)

    _all = category_encode(_all, cat_cols + one_hot_cols)

    train = _all[_all[is_train] == 1]
    test = _all[_all[is_train] == 0]

    train.drop(columns=[is_train], inplace=True)
    test.drop(columns=[is_train], inplace=True)

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

    # catboost_params = {
    #     'learning_rate': 0.01,
    #     'max_depth': 8,
    #     'bagging_temperature': 0.8,
    #     'l2_leaf_reg': 45,
    #     'od_type': 'Iter'
    # }
    #
    # fit_params = {
    #     "early_stopping_rounds": 100,
    #     "verbose": 5000
    # }
    #
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
