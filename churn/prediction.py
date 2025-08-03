import pickle
import pandas as pd

def load_model(model_path="models/customer_churn_model.pkl", enc_path="models/encoders.pkl"):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], encoders, model_data["features_names"]

def make_prediction(input_dict, model, encoders):
    input_df = pd.DataFrame([input_dict])
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])
    return model.predict(input_df), model.predict_proba(input_df)
