import pandas as pd


def built_year(df):
    df['BuildingYear'] = df['BuildingYear'].dropna()
    df['BuildingYear'] = df['BuildingYear'].str.replace('戦前', '昭和20年')
    df['年号'] = df['BuildingYear'].str[:2]
    df['和暦年数'] = df['BuildingYear'].str[2:].str.strip('年').fillna(0).astype(
        int)
    df.loc[df['年号'] == '昭和', 'BuildingYear'] = df['和暦年数'] + 1925
    df.loc[df['年号'] == '平成', 'BuildingYear'] = df['和暦年数'] + 1988
    df['BuildingYear'] = pd.to_numeric(df['BuildingYear'])
    return df


def walk_time(df):
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('30分?60分',
                                                                    '45')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('1H?1H30',
                                                                    '75')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('1H30?2H',
                                                                    '105')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('2H?',
                                                                    '120')
    df['TimeToNearestStation'] = pd.to_numeric(df['TimeToNearestStation'],
                                               errors='coerce')
    return df


def area1(df):
    replace_dict = {'10m^2未満': 9, '2000㎡以上': 2000}
    df['TotalFloorArea'] = pd.to_numeric(
        df['TotalFloorArea'].replace(replace_dict))
    return df


def area2(df):
    replace_dict = {'2000㎡以上': 2000, '5000㎡以上': 5000}
    df['Area'] = pd.to_numeric(df['Area'].replace(replace_dict))
    return df


def maguchi(df):
    df['Frontage'] = pd.to_numeric(df['Frontage'].replace('50.0m以上', 50.0))
    return df


def landshape(df):
    # ほぼ長方形 -> 長方形
    df["LandShape"] = df["LandShape"].replace("^ほぼ", "", regex=True)
    return df


def series_split_colum(df, col_name):
    split_df = df[col_name].str.get_dummies(sep="、")
    col_names = ["{}_{}".format(col_name, idx) for idx in
                 range(split_df.shape[1])]
    split_df.columns = col_names
    return df.join(split_df)


def structure(df):
    col_name = "Structure"
    """
    建物の構造: Structure

        ブロック造  木造  軽量鉄骨造  鉄骨造  ＲＣ  ＳＲＣ
    0           0   0      0    0   0    1
    1           0   0      0    0   1    0
    """
    return series_split_colum(df, col_name)


def remarks(df):
    col_name = "Remarks"
    """
    取引の事情等: Remarks

        その他事情有り  他の権利・負担付き  古屋付き・取壊し前提  ...
    0             0          0           0  ...
    1             0          0           0  ...
    """
    return series_split_colum(df, col_name)


def use(df):
    col_name = "Use"
    """
    用途: Use

        その他  事務所  住宅  作業場  倉庫  共同住宅  工場  店舗  駐車場
    0         0    0   0    0   0     0   0   0    0
    1         0    0   0    0   0     0   0   0    0
    2         0    0   1    0   0     0   0   0    0
    3         0    0   1    0   0     0   0   0    0
    4         0    1   1    0   0     0   0   1    0
    """
    return series_split_colum(df, col_name)
