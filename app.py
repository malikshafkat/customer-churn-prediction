# app.py

from churn.preprocessing import (
    load_data, clean_data, encode_target,
    encode_categorical_features, split_data, apply_smote
)
from churn.training import (
    get_models, cross_validate_models, train_final_model, save_model
)
from churn.evaluate import evaluate_model
from churn.prediction import predict_single

def main():
    # === 1. Load and preprocess data ===
    data_path = "data/Customer-Churn.csv"
    encoders_path = "encoders.pkl"
    model_path = "models/customer_churn_model.pkl"

    print("Loading data...")
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_target(df)
    df = encode_categorical_features(df, encoders_path)

    # Split and balance
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # === 2. Cross-validation ===
    print("\nCross-validating models...")
    models = get_models()
    cross_validate_models(models, X_train_bal, y_train_bal)

    # === 3. Train final model ===
    print("\nTraining final Random Forest model...")
    final_model = train_final_model(X_train_bal, y_train_bal)

    # Save the model
    save_model(final_model, X_train.columns.tolist(), model_path)

    # === 4. Evaluate on test data ===
    print("\nEvaluating model...")
    evaluate_model(final_model, X_test, y_test)

    # === 5. Example prediction ===
    print("\nTesting prediction system with a sample input...")
    sample_input = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    result, pred_prob = predict_single(sample_input, model_path=model_path, encoders_path=encoders_path)
    print(f"Prediction: {result}")
    print(f"Prediction Probabilities: {pred_prob}")

if __name__ == "__main__":
    main()
