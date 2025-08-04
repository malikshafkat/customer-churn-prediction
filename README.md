# Customer Churn Prediction

This project predicts if a telecom customer will leave (churn) using machine learning.

We used a CSV dataset, cleaned it, explored patterns, trained models, and built a prediction system.

---

## 1. Problem Definition

- **Goal:** Predict whether a customer will churn (leave the service).
- **Why important:** Companies can take action early and reduce losses.
- **Challenges:** Data is imbalanced, and many columns are categorical.

---

## 2. Data Cleaning

- Removed `customerID` column (not useful for prediction).
- Fixed wrong values in `TotalCharges` column (empty strings changed to 0).
- Converted `TotalCharges` to numeric.
- Checked for missing values.

**Result:** Clean dataset ready for analysis.

---

## 3. Exploratory Data Analysis (EDA)

We studied the dataset using:
- **Histograms** – to see the spread of `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Boxplots** – to find outliers.
- **Heatmap** – to see correlation between numeric columns.
- **Countplots** – for categorical columns like gender, contract type, payment method.

**Observation:**
- Customers with **month-to-month contracts and high charges** churn more.
- Tenure (length of time with company) has a strong effect: low tenure → high churn.

---

## 4. Data Preprocessing

- Changed target column `Churn` to 1 (Yes) and 0 (No).
- Used **Label Encoding** for all categorical columns.
- Split the data into **80% training and 20% testing**.
- Applied **SMOTE** to fix the class imbalance.

---

## 5. Model Training

We trained three models:
- Decision Tree
- Random Forest
- XGBoost

We used cross-validation to compare them.

**Observation:**  
Random Forest performed the best.

---

## 6. Model Evaluation

After training Random Forest, we tested it on the test data.

**Results:**
- **Accuracy:** Around 80% (from your notebook)
- **Confusion Matrix:** Showed balanced performance.
- **Classification Report:** Good precision and recall for both churn and non-churn classes.

---

## 7. Prediction System

- The final model is saved as a `.pkl` file.
- For a new customer, the system:
  - Encodes the data
  - Loads the saved model
  - Returns:
    - **Prediction:** `Churn` or `No Churn`
    - **Probability scores**

---

## 8. Complete Pipeline

The `app.py` script runs everything:
1. Loads and cleans the dataset
2. Prepares data
3. Trains models and selects the best one
4. Evaluates performance
5. Saves the final model
6. Makes a sample prediction

---

## How to Run

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
