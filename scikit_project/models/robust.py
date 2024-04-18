import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    df = pd.read_csv(ruta_csv)
    
    X = df.drop(['country', 'score'], axis= 1)
    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

    estimadores = {
        'SVR' : SVR(gamma = 'auto', C = 1.0, epsilon= 0.1),
        'RANSAC' : RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
    
        print("="*64)
        print(name)
        print(f"MSE: , {mean_absolute_error(y_test, predictions):.20f}")
    

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print(name)
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title(f'{name} VS Real')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()