import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Do NOT add a mlflow.set_tracking_uri() line here.
# MLflow will automatically use the MLFLOW_TRACKING_URI environment variable.

# Start an MLflow run
with mlflow.start_run() as run:
    # 1. Load the data
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define the model and its parameters
    C_value = 0.5
    model = LogisticRegression(C=C_value, solver="liblinear")

    # 4. Log the hyperparameters with MLflow
    mlflow.log_param("C", C_value)
    mlflow.log_param("solver", "liblinear")

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # 7. Log the metrics with MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 8. Log the model itself as an artifact
    mlflow.sklearn.log_model(model, "model")

    print(f"MLflow Run ID: {run.info.run_id}")

print("Training complete and metrics logged to MLflow.")