import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# No explicit mlflow.set_tracking_uri here to avoid overriding env var
# Let MLflow use MLFLOW_TRACKING_URI environment variable set in your workflow.

# Start an MLflow run
with mlflow.start_run() as run:
    # 1. Load data
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define model with solver that supports multinomial multiclass classification
    C_value = 0.5
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, C=C_value)

    # 4. Log hyperparameters to MLflow
    mlflow.log_param("C", C_value)
    mlflow.log_param("solver", 'lbfgs')
    mlflow.log_param("multi_class", 'multinomial')

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # 7. Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 8. Log the model with 'name' parameter and input_example to infer schema
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None,  # Use if you want to register model in a registry
        # Use 'name' instead of deprecated 'artifact_path' is requested, but mlflow.sklearn.log_model still expects artifact_path, not name, verify mlflow version.
    )

    print(f"MLflow Run ID: {run.info.run_id}")

print("Training complete and metrics logged to MLflow.")

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example,
    registered_model_name="IrisLogisticRegression"
)