import streamlit as st
import pandas as pd
import pickle

# Load model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn or stay.")

# Inputs

gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    # Apply encoders
    for col, encoder in encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    input_data = input_data[feature_names]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(
            f"Customer is likely to churn\n\nProbability: {probability[1]:.2%}"
        )
    else:
        st.success(
            f"Customer is likely to stay\n\nProbability: {probability[0]:.2%}"
        )

# About Section
st.markdown("---")

st.subheader("About This Project")

st.write("""
Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to continue using a company's service or leave it.

For Example : If a customer stops using a telecom company's services and moves to another provider, this is called **customer churn**.

Businesses use churn prediction to identify at-risk customers and take steps to improve customer satisfaction and retention.

In simple words, this model helps companies understand which customers may leave in the future so they can take action to keep them.
""")