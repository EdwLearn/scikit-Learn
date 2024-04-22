# Importar las bibliotecas necesarias
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

# Definir la función principal
if __name__ == "__main__":
    # Obtener el directorio principal del proyecto
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')

    # Leer el archivo csv y guardarlo en un DataFrame de pandas
    df_heart = pd.read_csv(ruta_csv)
    # Imprimir los primeros 5 registros del DataFrame
    print(df_heart.head(5))

    # Separar las variables independientes (X) y la variable dependiente (y)
    df_features = df_heart.drop(['target'], axis = 1)
    df_target = df_heart['target']

    # Estandarizar las variables independientes
    df_features = StandardScaler().fit_transform(df_features)

    # Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(df_features,df_target, test_size= 0.3, random_state=42)

    # Crear un objeto de la clase PCA con 3 componentes principales
    pca = PCA(n_components=3)
    # Ajustar el objeto PCA a los datos de entrenamiento
    pca.fit(X_train)

    # Crear un objeto de la clase IncrementalPCA con 3 componentes principales y un tamaño de lote de 10
    ipca = IncrementalPCA(n_components= 3, batch_size=10)
    # Ajustar el objeto IncrementalPCA a los datos de entrenamiento
    ipca.fit(X_train)

    # Crear un gráfico de la varianza explicada por cada componente principal
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    # Mostrar el gráfico
    plt.show()

    # Crear un objeto de la clase LogisticRegression
    logistic = LogisticRegression(solver='lbfgs')

    # Transformar los datos de entrenamiento y prueba con PCA
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    # Ajustar el modelo de regresión logística a los datos transformados con PCA
    logistic.fit(df_train, y_train)
    # Imprimir la precisión del modelo en los datos de prueba transformados con PCA
    print("SCORE PCA: ", logistic.score(df_test, y_test))

    # Transformar los datos de entrenamiento y prueba con IncrementalPCA
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    # Ajustar el modelo de regresión logística a los datos transformados con IncrementalPCA
    logistic.fit(df_train, y_train)
    # Imprimir la precisión del modelo en los datos de prueba transformados con IncrementalPCA
    print("SCORE IPCA: ", logistic.score(df_test, y_test))
    
'''
Estandariza las variables independientes para que tengan una media de 0 y una desviación estándar de 1. 
Esto es importante para muchos algoritmos de aprendizaje automático.
Divide el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba. 
El modelo se ajustará al conjunto de entrenamiento y se evaluará en el conjunto de prueba.
Crea un objeto de la clase PCA con 3 componentes principales y lo ajusta a los datos de entrenamiento. 
Luego, hace lo mismo con un objeto de la clase PCA Incremental.

PCA es una técnica que transforma las variables de un conjunto de datos en un nuevo conjunto de variables llamadas componentes principales. 
Estas componentes principales son combinaciones lineales de las variables originales y se calculan de manera que capturen la mayor cantidad de varianza posible
del conjunto de datos original. PCA requiere que todo el conjunto de datos esté en memoria para realizar el cálculo.

En contraste, IPCA es una variante de PCA que permite el procesamiento por lotes de los datos. 
Esto significa que puede manejar grandes conjuntos de datos que no caben en la memoria al procesar los datos en pequeños lotes. 
IPCA es útil cuando se trabaja con conjuntos de datos muy grandes, pero puede ser menos preciso que PCA.

'''
