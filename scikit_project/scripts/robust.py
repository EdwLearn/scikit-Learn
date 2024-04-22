# Importamos las librerías necesarias
import pandas as pd
import os
import matplotlib.pyplot as plt

# Importamos los modelos SVR, RANSACRegressor y HuberRegressor de sklearn
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR

# Importamos las funciones para dividir el conjunto de datos y calcular el error absoluto medio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Definimos la función principal
if __name__ == '__main__':
    # Obtenemos la ruta del directorio principal del proyecto
    project_root = os.getcwd()

    # Construimos la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    # Leemos el archivo csv con pandas
    df = pd.read_csv(ruta_csv)

    # Definimos nuestras variables independientes y dependiente
    X = df.drop(['country', 'score'], axis= 1)
    y = df['score']

    # Dividimos nuestros datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

    # Definimos los estimadores que vamos a utilizar
    estimadores = {
        'SVR' : SVR(gamma = 'auto', C = 1.0, epsilon= 0.1),
        'RANSAC' : RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    # Entrenamos cada estimador con nuestros datos de entrenamiento y realizamos predicciones con los datos de prueba
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)

        # Calculamos el error absoluto medio de cada modelo
        print("="*64)
        print(name)
        print(f"MSE: , {mean_absolute_error(y_test, predictions):.20f}")

    # Visualizamos las predicciones de cada modelo frente a los valores reales
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print(name)
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title(f'{name} VS Real')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()

'''
**Support Vector Regression (SVR):** Es una versión de las máquinas de vectores de soporte (SVM) que se utiliza para la regresión. 
A diferencia de la regresión lineal, que intenta minimizar el error cuadrado, SVR intenta ajustarse a los datos dentro de un cierto margen, denominado tubo de ε, 
y no se preocupa por los errores que están dentro de ese margen. Sin embargo, los errores fuera del margen son penalizados.

**RANSAC Regressor:** RANSAC (RANdom SAmple Consensus) es un algoritmo iterativo utilizado para estimar parámetros de un modelo matemático a partir de un conjunto de datos observados
que contienen valores atípicos. En el ámbito de la regresión, este modelo es útil cuando se espera que existan valores atípicos significativos en los datos.

**Huber Regressor:** El Regresor Huber es una variante de la regresión robusta que es menos sensible a los valores atípicos en los datos que la regresión lineal ordinaria.
Se utiliza cuando se desea una regresión que sea robusta a los valores atípicos, pero que todavía conserve las propiedades de un estimador de mínimos cuadrados cuando los datos están libres de valores atípicos.
'''