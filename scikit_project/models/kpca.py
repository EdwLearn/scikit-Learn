import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

if __name__ == "__main__":
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')

    df_heart = pd.read_csv(ruta_csv)
    # print(df_heart.head(5))
    
    # Tomando nuestra variable objetivo
    df_features = df_heart.drop(['target'], axis = 1)
    df_target = df_heart['target']
    
    # Cargar, ajustar y transformar nuestro modelo
    df_features = StandardScaler().fit_transform(df_features)
    
    # Conjunto de prueba y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(df_features,df_target, test_size= 0.3, random_state=42)
    
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)
    
    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)
    
    logistic = LogisticRegression(solver= 'lbfgs')
    logistic.fit(df_train, y_train)
    print("SCORE KPCA: ", logistic.score(df_test, y_test))