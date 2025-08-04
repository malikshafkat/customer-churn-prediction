import pickle
import pandas as pd

def load_model(model_path: str = "models/customer_churn_model.pkl"):
    """
    Load the trained model and feature names from a pickle file.
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["model"], model_data["features_names"]

def load_encoders(encoders_path: str = "encoders.pkl"):
    """
    Load saved label encoders.
    """
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return encoders

def prepare_input(input_data: dict, encoders: dict) -> pd.DataFrame:
    """
    Prepare a single row of input data for prediction.
    Encodes categorical columns using saved encoders.
    """
    input_df = pd.DataFrame([input_data])

    # Apply encoding for categorical features
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    return input_df

def predict_single(input_data: dict,
                   model_path: str = "models/customer_churn_model.pkl",
                   encoders_path: str = "encoders.pkl"):
    """
    Predict churn for a single input dictionary.
    Returns prediction label and probabilities.
    """
    model, feature_names = load_model(model_path)
    encoders = load_encoders(encoders_path)
    input_df = prepare_input(input_data, encoders)

    # Ensure input dataframe has same columns as training
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)
    pred_prob = model.predict_proba(input_df)

    result = "Churn" if prediction[0] == 1 else "No Churn"

    return result, pred_prob
