# Importamos librerias necesarias
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

# Verificar que el script se está ejecutando como programa principal
if __name__ == "__main__":
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')

    # Leer los datos del csv en un DataFrame
    df_heart = pd.read_csv(ruta_csv)
    # print(df_heart.head(5))

    # Tomar nuestras variables de interés, excluyendo la variable objetivo ('target')
    df_features = df_heart.drop(['target'], axis = 1)
    # Tomar nuestra variable objetivo
    df_target = df_heart['target']

    # Estandarizar las características usando StandardScaler
    df_features = StandardScaler().fit_transform(df_features)

    # Dividir el conjunto de datos en conjunto de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(df_features,df_target, test_size= 0.3, random_state=42)

    # Instanciar y entrenar el modelo KernelPCA con 4 componentes (dimensiones principales) y un kernel polinomial
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    # Transformar los conjuntos de entrenamiento y prueba con el modelo KernelPCA
    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    # Instanciar y entrenar el modelo LogisticRegression
    logistic = LogisticRegression(solver= 'lbfgs')
    logistic.fit(df_train, y_train)
    # Imprimir el score del modelo en el conjunto de prueba
    print("SCORE KPCA: ", logistic.score(df_test, y_test))


'''
PCA (Principal Component Analysis) es una técnica que se utiliza para resaltar la variación y sacar patrones fuertes en un conjunto de datos. 
Es muy útil cuando se tienen datos en dimensiones altas. Sin embargo, PCA puede no ser eficaz si los datos no son linealmente separables. 
KernelPCA, por otro lado, es una extensión de PCA que utiliza técnicas de kernel para transformar los datos en un espacio de mayor dimensión donde son linealmente separables antes de aplicar PCA. 
Esto puede ser útil cuando los patrones en los datos son complejos y no lineales.

StandardScaler es una función de preprocesamiento en sklearn que estandariza las características eliminando la media y escalando a la varianza unitaria. 
En otras palabras, cambia los datos a una distribución normal estándar con media 0 y desviación estándar 1. Esto es útil porque muchos algoritmos de aprendizaje automático,
como el PCA, asumen que todos los atributos están centrados alrededor de cero y tienen varianza en la misma orden.
'''