import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Do NOT set tracking URI here; let MLflow pick up MLFLOW_TRACKING_URI env var from GitHub Actions

with mlflow.start_run() as run:
    # 1. Load the data
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define the model with recommended solver and parameters
    C_value = 0.5
    model = LogisticRegression(solver="lbfgs", C=C_value, max_iter=1000)

    # 4. Log hyperparameters with MLflow
    mlflow.log_param("C", C_value)
    mlflow.log_param("solver", "lbfgs")

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # 7. Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 8. Log the model with input_example for signature inference
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print(f"MLflow Run ID: {run.info.run_id}")

print("Training complete and metrics logged to MLflow.")
