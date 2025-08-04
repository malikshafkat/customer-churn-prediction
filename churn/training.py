import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def get_models():
    """
    Returns a dictionary of models to train.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    return models

def cross_validate_models(models: dict, X_train, y_train, cv: int = 5):
    """
    Perform cross-validation for each model.
    Returns a dictionary of cross-validation scores.
    """
    cv_scores = {}
    for model_name, model in models.items():
        print(f"Training {model_name} with default parameters")
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[model_name] = scores
        print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
        print("-"*70)
    return cv_scores

def train_final_model(X_train, y_train):
    """
    Train the final model (Random Forest, as selected in the notebook).
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, feature_names, path: str = "models/customer_churn_model.pkl"):
    """
    Save the trained model and feature names using pickle.
    """
    model_data = {
        "model": model,
        "features_names": feature_names
    }
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
