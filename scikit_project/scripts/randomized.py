import pandas as pd
import os

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'felicidad.csv')
    df = pd.read_csv(ruta_csv)
    
    X = df.drop(['country', 'rank', 'score'], axis = 1)
    y = df[['score']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['squared_error', 'absolute_error'], #Medida de calidad de los splits
        'max_depth' : range(2, 11)
    }
    
    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv = 3, scoring='neg_mean_absolute_error').fit(X,y) # Con 'n_iter' toma n conjuntos distintos para hacer la iteración

    print(rand_est.best_estimator_)
    
    print(rand_est.best_params_)
    
    print(rand_est.predict(X.loc[[0]]))