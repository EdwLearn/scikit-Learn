# Importamos las bibliotecas necesarias
import pandas as pd
import os

# Algoritmos de clasificación
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

# Herramientas para dividir los datos y evaluar el rendimiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Otros algoritmos de clasificación
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Ignoramos las advertencias para limpiar la salida
import warnings
warnings.simplefilter("ignore")

# La ejecución del script empieza aquí
if __name__ == '__main__':
    # Obtenemos la ruta del directorio actual
    project_root = os.getcwd()

    # Creamos la ruta al archivo csv, que está en la carpeta 'data'
    ruta_csv = os.path.join(project_root, 'data', 'heart.csv')
    # Leemos el archivo csv en un DataFrame
    df_heart = pd.read_csv(ruta_csv)

    # Separamos las variables explicativas (X) de la variable objetivo (y)
    X = df_heart.drop(['target'], axis = 1)
    y = df_heart['target']

    # Dividimos los datos en conjuntos de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    # Entrenamos el algoritmo KNN y lo usamos para hacer predicciones
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)

    # Imprimimos la precisión del modelo KNN
    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    # Entrenamos un ensamble de tipo Bagging con KNN y lo usamos para hacer predicciones
    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)

    # Imprimimos la precisión del modelo Bagging
    print("="*64)
    print(accuracy_score(bag_pred, y_test))

    # Definimos un diccionario con varios algoritmos de clasificación
    classifier = {
        'KNeighborsClassifier' : KNeighborsClassifier(),
        'LinearSVC' : LinearSVC(),
        'SVC' : SVC(),
        'SGDC' : SGDClassifier(),
        'DecisionTreeClassifier' : DecisionTreeClassifier()
    }

    # Para cada algoritmo en el diccionario
    for name, estimador in classifier.items():
        # Entrenamos un modelo de Bagging con ese algoritmo
        bag_class = BaggingClassifier(estimator= estimador, n_estimators= 5).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)

        # Imprimimos la precisión del modelo Bagging con ese algoritmo
        print(f'Accuracy Bagging with {name}:', accuracy_score(bag_pred, y_test))
        print('')

'''  
K-Nearest Neighbors, o KNN, es un algoritmo de aprendizaje supervisado que se usa principalmente para problemas de clasificación, 
aunque también puede usarse en problemas de regresión. Es un algoritmo basado en instancias, lo que significa que no aprende explícitamente un modelo. 
En lugar de eso, clasifica nuevos casos basándose en su similitud con casos de entrenamiento existentes. Funciona de manera muy simple: 
un nuevo caso es clasificado por mayoría de votos de sus vecinos K más cercanos. 
Cada objeto se clasifica por sus vecinos más cercanos; 
por lo tanto, un objeto se asigna a la clase más común entre sus K vecinos más cercanos.
'''    

