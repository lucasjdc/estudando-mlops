import pandas as pd
import mlflow
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
df = pd.read_csv('casas.csv')
X = df.drop('preco', axis=1)
y = df['preco'].copy()

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configurar MLflow com SQlite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('house-prices-eda')

# Linear Regression
mlflow.start_run()
lr = LinearRegression()
lr.fit(X_train, y_train)

mlflow.sklearn.log_model(lr, 'lr')
lr_predict = lr.predict(X_test)
print(X_test.iloc[0])
print(y_test)

mse = mean_squared_error(y_test, lr_predict)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, lr_predict)

print("Regressão Linear")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

mlflow.log_metric('mse', mse)
mlflow.log_metric('rmse', rmse)
mlflow.log_metric('r2', r2)

mlflow.end_run()

# XGbosst
with mlflow.start_run():
	xgb = XGBRFRegressor(random_state=42)
	xgb.fit(X_train, y_train)
	mlflow.xgboost.log_model(xgb, 'xgboost')
	xgb_predicted = xgb.predict(X_test)
	mse = mean_squared_error(y_test, xgb_predicted)
	rmse = math.sqrt(mse)
	r2 = r2_score(y_test, xgb_predicted)
	print("XGBoost")
	print(f"RMSE: {rmse}")
	print(f"R2: {r2}")
	mlflow.log_metric('mse', mse)
	mlflow.log_metric('rmse', rmse)
	mlflow.log_metric('r2', r2)

# Para visualizar os experimentos, use no terminal:
#mlflow ui --backend-store-uri sqlite:///mlflow.db