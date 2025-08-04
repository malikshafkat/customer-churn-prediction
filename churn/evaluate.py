from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data and print metrics.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy Score:\n", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }
