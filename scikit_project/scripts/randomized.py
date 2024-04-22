# Importa las bibliotecas necesarias
import pandas as pd
import os

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Obtiene el directorio principal del proyecto
    project_root = os.getcwd()

    # Construye la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    # Lee el archivo csv y lo guarda en un DataFrame de pandas
    df = pd.read_csv(ruta_csv)

    # Separa las variables independientes (X) y la variable dependiente (y)
    X = df.drop(['country', 'rank', 'score'], axis = 1)
    y = df[['score']]

    # Crea un objeto de la clase RandomForestRegressor
    reg = RandomForestRegressor()

    # Define los parámetros para la búsqueda aleatorizada
    parametros = {
        'n_estimators' : range(4,16), # Número de árboles en el bosque
        'criterion' : ['squared_error', 'absolute_error'], # Medida de calidad de los splits
        'max_depth' : range(2, 11) # Profundidad máxima del árbol
    }

    # Realiza la búsqueda aleatorizada con validación cruzada
    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv = 3, scoring='neg_mean_absolute_error').fit(X,y)

    # Imprime el mejor estimador encontrado
    print(rand_est.best_estimator_)

    # Imprime los mejores parámetros encontrados
    print(rand_est.best_params_)

    # Predice el valor para el primer registro del DataFrame
    print(rand_est.predict(X.loc[[0]]))

'''
El error medio absoluto negativo es una métrica utilizada en problemas de regresión para medir la diferencia entre los valores predichos por el modelo y los valores reales. 
Es el promedio de las diferencias absolutas entre las predicciones y los valores reales. Sin embargo, en el caso de la métrica del error medio absoluto negativo, 
los valores obtenidos se multiplican por -1.

Esta métrica se utiliza comúnmente en la validación cruzada y en los procedimientos de selección de modelos, como la búsqueda aleatorizada de hiperparámetros. 
El motivo de utilizar el valor negativo en lugar del positivo es que estos procedimientos buscan maximizar la métrica de puntuación. En el caso del error medio absoluto, 
un valor más bajo es mejor, por lo que al tomar el negativo, un valor más alto (es decir, más cercano a 0) es mejor.
'''