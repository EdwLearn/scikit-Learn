# Importar las bibliotecas necesarias
import pandas as pd
import os
from sklearn.cluster import MeanShift

# Importar las bibliotecas necesarias para la visualización
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Obtener el directorio principal del proyecto
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'candy.csv')
    # Leer el archivo csv y guardarlo en un DataFrame de pandas
    df_candy = pd.read_csv(ruta_csv)

    # Separar las variables independientes (X)
    X = df_candy.drop('competitorname', axis = 1)

    # Crear un objeto de la clase MeanShift y ajustarlo a los datos
    meanshift = MeanShift().fit(X)
    # Imprimir el número máximo de etiquetas (clusters)
    print(max(meanshift.labels_))
    print('='*64)

    # Imprimir los centros de los clusters
    print(meanshift.cluster_centers_)
    print('='*64)

    # Añadir las etiquetas (clusters) al DataFrame original
    df_candy['meanshift'] = meanshift.labels_
    # Imprimir el DataFrame con las etiquetas añadidas
    print(df_candy)

    # Crear un objeto de la clase PCA y ajustarlo a los datos
    pca = PCA(n_components=2)
    pca.fit(X)
    # Transformar los datos a 2 dimensiones
    pca_data = pca.transform(X)

    # Crear un nuevo objeto de la clase MeanShift y ajustarlo a los datos transformados
    meanshift = MeanShift().fit(pca_data)
    # Crear un gráfico de dispersión con los datos transformados y colorearlos según las etiquetas
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=meanshift.predict(pca_data))
    # Añadir los centros de los clusters al gráfico de dispersión
    plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='black', s=150)
    # Mostrar el gráfico
    plt.show()
    
'''
MeanShift es un algoritmo de clustering (agrupamiento) no paramétrico basado en densidad. 
Se utiliza para detectar la estructura espacial de los datos, como la presencia de grupos o clusters.

Un aspecto importante de MeanShift es que no requiere especificar el número de clusters a priori, lo cual es una ventaja en situaciones en las que no se conoce esta información. 
Sin embargo, la elección del ancho de banda (un parámetro que determina la "vecindad" de los puntos) puede influir en la cantidad de clusters que el algoritmo identifica.
'''
