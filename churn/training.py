import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_and_save_model(df, output_model_path, output_enc_path):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)

    with open(output_model_path, "wb") as f:
        pickle.dump({"model": model, "features_names": X.columns.tolist()}, f)
    with open(output_enc_path, "wb") as f:
        pickle.dump(encoders, f)
