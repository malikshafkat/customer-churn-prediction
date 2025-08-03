import pandas as pd

def clean_data(df):
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    return df
