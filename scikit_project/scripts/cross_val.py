import pandas as pd
import os
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    df = pd.read_csv(ruta_csv)
    
    X = df.drop(['country', 'score'], axis = 1)
    y = df['score']

    model = DecisionTreeRegressor()

    score = cross_val_score(model, X, y, cv =3,scoring='neg_mean_squared_error') #Validación rápida de configuración por defecto
    print(score)
    print("="*32)
    print(np.mean(score))
    print("="*32)
    print(np.abs(np.mean(score)))
    kf = KFold(n_splits=3, shuffle= True, random_state=42)

    for train, test in kf.split(df):
        print(train)
        print(test)
        print('\n')
    
    X = df.drop(['country', 'score'], axis=1)
    y = df['score']

    print(df.shape)

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X,y, cv=3, scoring='neg_mean_squared_error') # con cv podemos controlar el numOfFolds
    print(score)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_values = []

    for train, test in kf.split(df):
        print(train)
        print(test)

        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]


        model = DecisionTreeRegressor().fit(X_train, y_train)
        predict = model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, predict))

    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))