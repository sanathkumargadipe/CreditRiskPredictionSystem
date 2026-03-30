import pandas as pd

TOP_FEATURES = [
    "credit_history",
    "amount",
    "duration",
    "age",
    "employment_duration",
    "savings",
    "purpose",
    "other_debtors"
]

def select_features(df):

    X = df[TOP_FEATURES]

    y = df["risk"]

    return X, y