import pandas as pd


def convert_strings_to_one_hot(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    for col in df:
        if type(df[col].iloc[0]) == str:
            df = pd.concat([df, pd.get_dummies(df[col])], axis=1).drop(col, axis=1)
    return df
