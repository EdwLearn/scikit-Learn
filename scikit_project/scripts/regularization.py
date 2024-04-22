# Importamos las librerías necesarias
import pandas as pd
import os
import sklearn

# Importamos los modelos de regresión lineal, Lasso y Ridge de sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Importamos las funciones para dividir el conjunto de datos y calcular el error cuadrado medio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Definimos la función principal
if __name__ == '__main__':
    # Obtenemos la ruta del directorio principal del proyecto
    project_root = os.getcwd()

    # Construimos la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    # Leemos el archivo csv con pandas
    dataset = pd.read_csv(ruta_csv)

    # Definimos nuestras variables independientes y dependiente
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    # Imprimimos la forma de nuestras variables para verificar que todo está bien
    print(X.shape)
    print(y.shape)

    # Dividimos nuestros datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    # Entrenamos un modelo de regresión lineal, Lasso y Ridge con nuestros datos de entrenamiento
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    # Calculamos el error cuadrado medio de cada modelo
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    # Imprimimos los coeficientes de los modelos Lasso y Ridge
    print("="*32)
    print("Coef Lasso")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)


'''
Lasso (Least Absolute Shrinkage and Selection Operator) es una técnica de regresión que realiza tanto la selección de variables como la regularización con el fin de mejorar 
la precisión y la interpretabilidad del modelo estadístico que produce. Lasso tiende a hacer que los coeficientes de las variables menos importantes sean cero, efectivamente eliminándolas del modelo.

Ridge es una técnica de regresión para analizar datos multicolineales, una situación en la que las variables independientes están altamente correlacionadas. En términos generales, 
el método Ridge es una regularización del método de mínimos cuadrados ordinarios, que minimiza la suma de los cuadrados de los residuos pero con una restricción que limita la suma de los cuadrados de los coeficientes.

La principal diferencia entre los tres es cómo tratan a las variables correlacionadas. La regresión lineal las mantiene, Lasso tiende a eliminar una de las dos y mantener la otra, 
mientras que Ridge las mantiene a ambas pero reduce sus coeficientes.

El Error Cuadrado Medio (Mean Squared Error, MSE) es una medida de cuán cerca están las predicciones de un modelo de los valores reales. Es la media de los cuadrados de las diferencias entre los valores predichos 
y los valores reales. Un MSE más pequeño indica que las predicciones del modelo están más cerca de los valores reales.
'''