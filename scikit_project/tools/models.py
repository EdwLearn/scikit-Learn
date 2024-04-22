# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np

# Importa las clases de modelos de regresión de sklearn
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
# Importa la clase para la búsqueda en malla de sklearn
from sklearn.model_selection import GridSearchCV

# Importa la clase Utils de la carpeta tools
from tools.utils import Utils

# Define la clase Model
class Model:
    # Define el constructor de la clase
    def __init__(self):
        # Inicializa los modelos de regresión
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }

        # Define los parámetros para la búsqueda en malla
        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['auto', 'scale'],
                'C': [1,5,10]
            },
            'GRADIENT': {
                'loss' : ['squared_error','absolute_error'],
                'learning_rate' : [0.01, 0.05, 0.1]
            }

        }

    # Define el método para realizar la búsqueda en malla y entrenar los modelos
    def grid_training(self, X, y):

        # Inicializa las variables para el mejor puntaje y el mejor modelo
        best_score = 999
        best_model = None

        # Itera sobre los modelos de regresión
        for name, reg in self.reg.items():

            # Realiza la búsqueda en malla y entrena el modelo
            grid_reg = GridSearchCV(reg, self.params[name], cv = 3).fit(X, y.values.ravel())
            # Calcula el puntaje absoluto del mejor modelo encontrado
            score = np.abs(grid_reg.best_score_)

            # Si el puntaje es mejor que el mejor puntaje hasta ahora, actualiza el mejor puntaje y el mejor modelo
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        # Crea un objeto de la clase Utils
        utils = Utils()
        # Exporta el mejor modelo y su puntaje
        utils.model_export(best_model, best_score)
