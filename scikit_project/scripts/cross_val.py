# Importamos las bibliotecas necesarias
import pandas as pd
import os
import numpy as np

# Importamos el modelo de regresión de árbol de decisión
from sklearn.tree import DecisionTreeRegressor

# Importamos las herramientas para la validación cruzada y el cálculo del error cuadrático medio
from sklearn.model_selection import (
    cross_val_score, KFold
)

from sklearn.metrics import mean_squared_error

# La ejecución del script empieza aquí
if __name__ == '__main__':
    project_root = os.getcwd()

    # Construimos la ruta al archivo csv, que está en la carpeta 'data'
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    # Leemos el archivo csv en un DataFrame
    df = pd.read_csv(ruta_csv)

    # Separamos las variables explicativas (X) de la variable objetivo (y)
    X = df.drop(['country', 'score'], axis = 1)
    y = df['score']

    # Creamos un modelo de regresión de árbol de decisión
    model = DecisionTreeRegressor()

    # Calculamos el error cuadrático medio negativo del modelo usando validación cruzada
    score = cross_val_score(model, X, y, cv =3, scoring='neg_mean_squared_error')
    print(score)
    print("="*32)

    # Calculamos la media del error cuadrático medio negativo
    print(np.mean(score))
    print("="*32)

    # Calculamos el valor absoluto de la media del error cuadrático medio negativo
    print(np.abs(np.mean(score)))

    # Creamos un generador de particiones para la validación cruzada
    kf = KFold(n_splits=3, shuffle= True, random_state=42)

    # Imprimimos los índices de los conjuntos de entrenamiento y prueba para cada partición
    for train, test in kf.split(df):
        print(train)
        print(test)
        print('\\n')

    # Volvemos a separar las variables explicativas (X) de la variable objetivo (y)
    X = df.drop(['country', 'score'], axis=1)
    y = df['score']

    # Imprimimos la forma del DataFrame
    print(df.shape)

    # Creamos otro modelo de regresión de árbol de decisión
    model = DecisionTreeRegressor()

    # Calculamos otra vez el error cuadrático medio negativo del modelo usando validación cruzada
    score = cross_val_score(model, X,y, cv=3, scoring='neg_mean_squared_error')
    print(score)
    print(np.abs(np.mean(score)))

    # Creamos otro generador de particiones para la validación cruzada
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_values = []

    # Para cada partición, entrenamos el modelo en el conjunto de entrenamiento, hacemos predicciones en el conjunto de prueba, 
    # y calculamos el error cuadrático medio
    for train, test in kf.split(df):
        print(train)
        print(test)

        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]


        model = DecisionTreeRegressor().fit(X_train, y_train)
        predict = model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, predict))

    # Imprimimos los tres errores cuadráticos medios y su media
    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))
    
'''
El modelo de regresión de Árbol de Decisión es un tipo de algoritmo de aprendizaje supervisado que se utiliza principalmente para problemas de regresión, 
aunque también puede usarse para problemas de clasificación. Este modelo predice el valor de la variable objetivo mediante el aprendizaje de reglas de decisión simples 
inferidas de las características de los datos.

En cuanto al Error Cuadrático Medio (ECM), es una métrica que se utiliza para medir la cantidad de error que hay entre dos conjuntos de datos. 
En otras palabras, es una medida de la calidad de un estimador, cuantifica el promedio de los cuadrados de los errores, es decir, 
la diferencia entre el estimador y lo que se estima.

El ECM se calcula como el promedio de los errores al cuadrado, de ahí que se llame "cuadrático". Por lo tanto, si el ECM es negativo, 
significa que el modelo está muy lejos de la línea de regresión perfecta.

El motivo por el que el ECM puede ser negativo en la validación cruzada es porque Scikit-Learn invierte los signos de las puntuaciones de los errores (como el ECM) 
debido a su convención de que las puntuaciones de los modelos deben ser maximizadas, no minimizadas. Por lo tanto, un ECM más grande (más positivo) 
es peor que un ECM más pequeño (más negativo), por lo que Scikit-Learn lo convierte en un número negativo para indicar que es peor.

Se imprime este valor, su media y su valor absoluto para tener una visión más completa del rendimiento del modelo. 
El valor individual del ECM nos da una idea de cuán lejos están nuestras predicciones del valor real, la media del ECM nos proporciona un resumen de estos errores 
en todas las predicciones y el valor absoluto del ECM nos permite interpretar este valor sin tener en cuenta la dirección del error.
'''