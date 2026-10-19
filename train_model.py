"""Regenerate customer_churn_model.pkl + encoders.pkl.

Replicates Customer_Churn_Prediction.ipynb preprocessing exactly, with one
correction: the final Random Forest is fitted on the SMOTE-resampled training
data (X_train_smote, y_train_smote) — the data the benchmark cross-validation
was run on — instead of the raw imbalanced split. Effect on the held-out test
set: accuracy 0.795 -> 0.772, churn-class recall 0.47 -> 0.57 (the model now
catches ~57% of churners instead of under half).

Run:  python train_model.py
"""
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    encoders = {}
    for column in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)

    predictions = model.predict(X_test)
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

    with open("customer_churn_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features_names": X.columns.tolist()}, f)
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("Saved customer_churn_model.pkl and encoders.pkl")


if __name__ == "__main__":
    main()
