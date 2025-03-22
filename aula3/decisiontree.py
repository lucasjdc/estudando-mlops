import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlops_intro")

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.rand(100, 1) * 2

print(f"Primeiro 5 exemplos de X:\n{X[:5]}")
print(f"Primeiro 5 exemplos de y:\n{y[:5]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "RandomForestRegressor")

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"Modelo RandomForest registrado no MLflow! Run ID:  {run.info.run_id}")