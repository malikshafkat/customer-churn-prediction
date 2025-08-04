import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset:
    - Drops customerID column
    - Fixes TotalCharges column
    """
    # Drop customerID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Replace empty strings in TotalCharges with 0.0 and convert to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
        df["TotalCharges"] = df["TotalCharges"].astype(float)

    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target column 'Churn' (Yes -> 1, No -> 0).
    """
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    return df

def encode_categorical_features(df: pd.DataFrame, encoders_path: str = "encoders.pkl") -> pd.DataFrame:
    """
    Label encode all categorical (object) columns and save encoders.
    """
    object_columns = df.select_dtypes(include="object").columns
    encoders = {}

    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder

    # Save encoders
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)

    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and test sets.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Applies SMOTE to balance the training dataset.
    """
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)
