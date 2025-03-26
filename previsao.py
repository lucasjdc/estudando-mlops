import pandas as pd
import mlflow
import xgboost
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def parse_args():
	parser = argparse.ArgumentParser(description="House Prices ML")
	parser.add_argument(
		'--learning-rate',
		type=float,
		default=0.3,
		help="Taxa de aprendizado atualizar o tamanho de cada passo do boosting"
	)
	parser.add_argument(
		'--max-depth',
		type=int,
		default=6,
		help="Profundidade maxima das arvores"
	)
	return parser.parse_args()

# Carregar dados
df = pd.read_csv('casas.csv')
X = df.drop('preco', axis=1)
y = df['preco'].copy()

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configurar MLflow com SQlite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('house-prices-eda')

# XGbosst

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

def main():
	args = parse_args()
	xgb_params = {
		'learning_rate': args.learning_rate,
		'max_depth': args.max_depth,
		'seed':42
	}

	with mlflow.start_run():
		mlflow.xgboost.autolog()
		evals = [(dtrain, 'train')]
		xgb = xgboost.train(xgb_params, dtrain, evals=evals)
		mlflow.xgboost.log_model(xgb, 'xgboost')
		xgb_predicted = xgb.predict(dtest)
		mse = mean_squared_error(y_test, xgb_predicted)
		rmse = math.sqrt(mse)
		r2 = r2_score(y_test, xgb_predicted)
		mlflow.log_metric('mse', mse)
		mlflow.log_metric('rmse', rmse)
		mlflow.log_metric('r2', r2)

if __name__ == "__main__":
	main()


# Para visualizar os experimentos, use no terminal:
#mlflow ui --backend-store-uri sqlite:///mlflow.db