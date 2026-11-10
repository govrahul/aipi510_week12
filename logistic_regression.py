import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow

mlflow.set_experiment("Logistic Regression Experiment")

RANDOM_STATE = 510

train_path = "Data/train.csv"

train_data = pd.read_csv(train_path)

X = train_data.drop(columns=["Potability"])
y = train_data["Potability"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

model = LogisticRegression(random_state=RANDOM_STATE)

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")