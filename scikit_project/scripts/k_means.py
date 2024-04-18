import pandas as pd
import os

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'candy.csv')
    df = pd.read_csv(ruta_csv)
    
    
    X = df.drop('competitorname', axis = 1)

    kmeans = KMeans(n_clusters= 4).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))
    
    print("="*64)
    
    print(kmeans.predict(X))
    
    df['group'] = kmeans.predict(X)
    
    # Gráfico de dispersión para dos características
    sns.scatterplot(x='winpercent',y='sugarpercent', hue='group', data=df, palette='viridis')
    plt.title('Clustering de golosinas basado en Azucar y su porcentaje de Victorias')
    plt.xlabel('Porcentaje de victorias')
    plt.ylabel('Contiene azucar')
    plt.legend(title='Grupo')
    plt.show()
    
    
    #Usando MiniBatchKMeans para experimentar los resultados
    X = df.drop(['competitorname','winpercent'], axis = 1)

    kmeans = MiniBatchKMeans(n_clusters= 4, batch_size= 50).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))
    
    df['group'] = kmeans.predict(X)
    
    sns.pairplot(df[['sugarpercent', 'pricepercent', 'winpercent', 'group']], hue='group')
    plt.show()
    
    
    # Graficando usando PCA
    
    X['cluster'] = kmeans.predict(X)

    model_PCA = PCA(n_components=2) # instanciamos el modelo 
    data_PCA = model_PCA.fit_transform(X) # realizamos tranformacion
    df_PCA = pd.DataFrame(data= data_PCA, columns= ["PCA1","PCA2"])
    df_PCA = pd.concat([df_PCA, X['cluster']], axis=1)

    x = sns.scatterplot(
        x = "PCA1", 
        y = "PCA2", 
        hue = "cluster", 
        data = df_PCA, 
        palette = ["red", "green", "blue", "black"]
    )
    plt.show()