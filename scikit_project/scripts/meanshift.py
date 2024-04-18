import pandas as pd
import os

from sklearn.cluster import MeanShift


if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'candy.csv')
    df_candy = pd.read_csv(ruta_csv)
    
    X = df_candy.drop('competitorname', axis = 1)

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    print('='*64)
    
    print(meanshift.cluster_centers_)
    print('='*64)
    
    df_candy['meanshift'] = meanshift.labels_
    print(df_candy)
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    pca.fit(X)
    pca_data = pca.transform(X)
    
    meanshift = MeanShift().fit(pca_data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=meanshift.predict(pca_data))
    plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='black', s=150)
    plt.show()
    