# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave (churn) or stay based on customer demographics, account information, and service usage patterns.

## Live Demo

🔗 Streamlit App:
https://customer-churn-prediction-2ydsrssrvaql6dqkexoj98.streamlit.app/

## Project Overview

Customer churn prediction helps businesses identify customers who are at risk of leaving their services. By predicting churn in advance, companies can take proactive measures to improve customer retention and reduce revenue loss.

## Dataset

**Telco Customer Churn Dataset**

### Features

* Customer Information (Gender, Senior Citizen, Partner, Dependents)
* Service Information (Phone Service, Internet Service, Tech Support, Streaming Services, etc.)
* Account Information (Tenure, Contract Type, Payment Method, Monthly Charges, Total Charges)

**Target Variable:** Churn (Yes / No)

## Project Workflow

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Label Encoding
* Handling Class Imbalance using SMOTE
* Model Training and Evaluation
* Model Persistence using Pickle
* Streamlit Web Application Development

## Models Trained

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

## Final Model

**Random Forest Classifier** was selected for deployment based on its performance and reliability.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn
* Streamlit
* Pickle

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
└── dataset/
```

## Installation

```bash
git clone https://github.com/raghuvendra34/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

pip install -r requirements.txt

streamlit run app.py
```

## Author

**Raghuvendra Kumar**

GitHub: https://github.com/raghuvendra34