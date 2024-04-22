import pandas as pd
import os

# Importamos las librerías de sklearn para clustering y reducción de dimensionalidad
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA

# Importamos las librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Obtenemos la ruta del directorio del proyecto
    project_root = os.getcwd()

    # Construimos la ruta del archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'candy.csv')
    # Leemos los datos del csv en un DataFrame
    df = pd.read_csv(ruta_csv)

    # Seleccionamos las columnas que nos interesan para el análisis y descartamos la columna 'competitorname'
    X = df.drop('competitorname', axis = 1)

    # Aplicamos el algoritmo de KMeans para dividir los datos en 4 grupos
    kmeans = KMeans(n_clusters= 4).fit(X)
    # Imprimimos el número de clusters
    print('Total de centros: ', len(kmeans.cluster_centers_))

    # Mostramos la clasificación de los datos
    print("="*64)
    print(kmeans.predict(X))

    # Añadimos la clasificación al DataFrame original
    df['group'] = kmeans.predict(X)

    # Creamos un gráfico de dispersión para visualizar los grupos en función de 'winpercent' y 'sugarpercent'
    sns.scatterplot(x='winpercent',y='sugarpercent', hue='group', data=df, palette='viridis')
    plt.title('Clustering de golosinas basado en Azucar y su porcentaje de Victorias')
    plt.xlabel('Porcentaje de victorias')
    plt.ylabel('Contiene azucar')
    plt.legend(title='Grupo')
    plt.show()

    # Ahora utilizamos MiniBatchKMeans, una variante de KMeans, para experimentar los resultados
    X = df.drop(['competitorname','winpercent'], axis = 1)

    # Entrenamos el modelo con nuestros datos
    kmeans = MiniBatchKMeans(n_clusters= 4, batch_size= 50).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))

    # Añadimos la clasificación al DataFrame original
    df['group'] = kmeans.predict(X)

    # Creamos un gráfico de pares para visualizar las relaciones entre las diferentes características
    sns.pairplot(df[['sugarpercent', 'pricepercent', 'winpercent', 'group']], hue='group')
    plt.show()

    # Graficamos los resultados usando PCA para reducir la dimensionalidad
    # Asignamos los clusters a una nueva columna en los datos
    X['cluster'] = kmeans.predict(X)

    # Instanciamos el modelo PCA con 2 componentes
    model_PCA = PCA(n_components=2)
    # Realizamos la transformación PCA en los datos
    data_PCA = model_PCA.fit_transform(X)
    # Creamos un nuevo DataFrame con los resultados de la transformación PCA
    df_PCA = pd.DataFrame(data= data_PCA, columns= ["PCA1","PCA2"])
    df_PCA = pd.concat([df_PCA, X['cluster']], axis=1)

    # Creamos un gráfico de dispersión de los resultados de PCA, coloreando los puntos según su cluster
    x = sns.scatterplot(
        x = "PCA1",
        y = "PCA2",
        hue = "cluster",
        data = df_PCA,
        palette = ["red", "green", "blue", "black"]
    )
    plt.show()

'''
- Un **cluster** en análisis de datos se refiere a un grupo de datos que comparten características similares. 
El clustering es una técnica de aprendizaje no supervisado que se utiliza para agrupar datos no etiquetados según sus similitudes. 
Es útil en muchas aplicaciones, incluyendo análisis de mercado, procesamiento de imágenes y minería de texto.


- **KMeans** y **MiniBatchKMeans** son algoritmos de clustering. 
KMeans es un algoritmo que divide un conjunto de N muestras en K clusters no solapados, cada uno de ellos descrito por la media de las muestras en el cluster. 
MiniBatchKMeans es una variante del algoritmo KMeans que utiliza mini-batches para reducir el tiempo de cálculo. 
Los mini-batches son subconjuntos del dataset de entrada que se toman al azar en cada iteración del algoritmo de optimización. 
En general, MiniBatchKMeans es más rápido pero puede dar resultados ligeramente peores que el KMeans tradicional.


- **PCA** (Principal Component Analysis) es una técnica de reducción de la dimensionalidad que se utiliza para transformar datos multidimensionales a 2 o 3 dimensiones 
para poder visualizarlos. PCA identifica las características más importantes de los datos y elimina las menos importantes, 
permitiendo visualizar los datos de una forma más simplificada sin perder demasiada información.


- Se crea un **nuevo dataframe con los resultados** para facilitar el análisis y la visualización de los datos después de la transformación. 
En este caso, el nuevo dataframe contiene los resultados de la transformación PCA, 
lo que permite visualizar fácilmente cómo se agrupan los datos en el espacio bidimensional.
'''