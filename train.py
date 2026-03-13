import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data/loan_data.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# SAVE FEATURE NAMES
joblib.dump(X.columns.tolist(), "features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")

    print("Accuracy:", acc)