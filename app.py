import streamlit as st
from churn.prediction import load_model, make_prediction

model, encoders, features = load_model()

st.title("Customer Churn Prediction")

user_input = {}
for feature in features:
    if feature in encoders:
        user_input[feature] = st.selectbox(feature, encoders[feature].classes_)
    else:
        user_input[feature] = st.number_input(feature)

if st.button("Predict"):
    pred, prob = make_prediction(user_input, model, encoders)
    st.write(f"Prediction: {'Churn' if pred[0] == 1 else 'No Churn'}")
    st.write(f"Probability: {prob}")
