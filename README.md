# Customer Churn Prediction

## Overview

Customer churn prediction is a machine learning project that predicts whether a telecom customer is likely to leave the service based on their demographic information, account details, and service usage patterns.

The goal is to help businesses identify customers at risk of churning and improve customer retention strategies.

---

## Dataset

Dataset: Telco Customer Churn Dataset

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

Target Variable:

- Churn (Yes / No)

---

## Project Workflow

### 1. Data Loading and Understanding
- Loaded telecom customer dataset
- Checked dataset dimensions and feature information

### 2. Data Cleaning
- Removed Customer ID column
- Handled missing values in TotalCharges
- Converted data types

### 3. Exploratory Data Analysis (EDA)
- Distribution analysis using histograms
- Outlier detection using boxplots
- Correlation heatmap for numerical features
- Categorical feature analysis

### 4. Data Preprocessing
- Label Encoding for categorical features
- Feature-target separation

### 5. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique)

### 6. Model Training
Trained the following machine learning models:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### 7. Model Evaluation
Evaluation metrics used:

- Accuracy Score
- Confusion Matrix
- Classification Report
- 5-Fold Cross Validation

### 8. Model Persistence
- Saved trained model using Pickle
- Saved encoders for future predictions

### 9. Predictive System
- Built a prediction pipeline for new customer data

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- Pickle
- Jupyter Notebook

---

## Results

The project compares multiple classification models and evaluates their performance using cross-validation and test-set metrics.

Random Forest was selected as the final model for prediction and deployment.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── customer_churn_model.pkl
├── encoders.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Future Improvements

- Streamlit Web Application
- Cloud Deployment
- Hyperparameter Tuning
- Feature Importance Visualization

---

## Author

Raghuvendra Kumar