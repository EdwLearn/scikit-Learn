import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    project_root = os.getcwd()

    # Construir la ruta al archivo csv relativa al directorio principal del proyecto
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')
    df_heart = pd.read_csv(ruta_csv)
    
    X = df_heart.drop(['target'], axis = 1)
    y = df_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)

    print("="*64)
    print(accuracy_score(knn_pred, y_test))
    
    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))

    classifier = {
        'KNeighborsClassifier' : KNeighborsClassifier(),
        'LinearSVC' : LinearSVC(),
        'SVC' : SVC(),
        'SGDC' : SGDClassifier(),
        'DecisionTreeClassifier' : DecisionTreeClassifier()
    }

    for name, estimador in classifier.items():
        bag_class = BaggingClassifier(estimator= estimador, n_estimators= 5).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)
    
        print(f'Accuracy Bagging with {name}:', accuracy_score(bag_pred, y_test))
        print('')
