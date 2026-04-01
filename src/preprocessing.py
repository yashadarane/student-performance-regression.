import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean(numeric_only=True))


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
            "Yes": 1,
            "No": 0
        })
    return df


def select_features(df, features, target):
    X = df[features]
    y = df[target]
    return X, y