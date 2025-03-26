import mlflow
logged_model = '/home/lucas/Alura/ciclo-vidas-vidas-modelos/mlruns/1/cdc3b9a51cc84c839e91567622a61ff8/artifacts/model'
#logged_model = 'runs:/cdc3b9a51cc84c839e91567622a61ff8/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('casas_X.csv')
predicted = loaded_model.predict(pd.DataFrame(data))

data['predicted'] = predicted
data.to_csv('precos.csv')