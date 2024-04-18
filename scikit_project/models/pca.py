import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

if __name__ == "__main__":
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')

    df_heart = pd.read_csv(ruta_csv)
    print(df_heart.head(5))
    
    # Tomando nuestra variable objetivo
    df_features = df_heart.drop(['target'], axis = 1)
    df_target = df_heart['target']
    
    # Cargar, ajustar y transformar nuestro modelo
    df_features = StandardScaler().fit_transform(df_features)
    
    # Conjunto de prueba y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(df_features,df_target, test_size= 0.3, random_state=42)
    
    #print(X_train.shape)
    #print(y_train.shape)
    
    # n_components = min(n_muestras, n_feautures)
    pca = PCA(n_components=3)
    pca.fit(X_train)
    
    ipca = IncrementalPCA(n_components= 3, batch_size=10)
    ipca.fit(X_train)
    
    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()
    
    logistic = LogisticRegression(solver='lbfgs')
    
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("SCORE PCA: ", logistic.score(df_test, y_test))
    
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("SCORE IPCA: ", logistic.score(df_test, y_test))
