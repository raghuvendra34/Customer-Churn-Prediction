# Customer Churn Prediction

## Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a telecom customer is likely to leave a company's service based on customer demographics, account information, and service usage patterns.

The objective of this project is to help businesses identify customers who are at risk of leaving and improve customer retention strategies through data-driven decision-making.

---

## Problem Statement

Customer retention is one of the most important challenges for subscription-based businesses. Acquiring a new customer is often more expensive than retaining an existing one.

This project uses Machine Learning techniques to analyze customer behavior and predict whether a customer is likely to churn.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer demographic information, subscribed services, account details, and churn status.

### Features

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### Target Variable

* Churn (Yes / No)

---

## Project Workflow

### 1. Data Loading and Understanding

* Loaded telecom customer dataset
* Explored dataset structure and dimensions
* Analyzed feature types

### 2. Data Cleaning

* Removed unnecessary columns
* Handled missing values
* Converted data types where required

### 3. Exploratory Data Analysis (EDA)

* Distribution analysis
* Boxplots for outlier detection
* Correlation heatmaps
* Categorical feature analysis

### 4. Data Preprocessing

* Label Encoding of categorical features
* Feature engineering and preparation
* Feature-target separation

### 5. Handling Class Imbalance

Applied:

* SMOTE (Synthetic Minority Oversampling Technique)

to balance the minority class and improve model performance.

### 6. Model Training

The following Machine Learning models were trained and evaluated:

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

### 7. Model Evaluation

Performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* 5-Fold Cross Validation

### 8. Model Persistence

Saved:

* Trained model using Pickle
* Label encoders for future predictions

### 9. Predictive System

Developed a prediction pipeline capable of generating predictions for new customer data.

---

## Streamlit Web Application

An interactive Streamlit application has been developed for real-time customer churn prediction.

### Features

* User-friendly interface
* Real-time predictions
* Churn probability estimation
* Customer retention insights
* Interactive customer data input

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Imbalanced-Learn
* Pickle
* Streamlit
* Jupyter Notebook

---

## Results

The project compares multiple machine learning models and evaluates their performance using cross-validation and test-set metrics.

**Random Forest Classifier** was selected as the final model for prediction and deployment due to its strong predictive performance and reliability.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
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

## Installation

Clone the repository:

```bash
git clone https://github.com/raghuvendra34/Customer-Churn-Prediction.git
```

Move into the project directory:

```bash
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Hyperparameter Tuning
* Feature Importance Visualization
* Advanced Customer Analytics Dashboard
* Real-Time Prediction API
* Model Monitoring and Retraining Pipeline

---

## Author

**Raghuvendra Kumar**
