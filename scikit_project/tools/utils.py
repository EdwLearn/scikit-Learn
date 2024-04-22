import pandas as pd
import joblib
import os

class Utils:
    
    # Cargar los datos .csv
    def load_from_csv(self, path):
        return pd.read_csv(path)
    
    # Cargar Datos de Base de Datos
    def load_from_mysql(self):
        pass
    
    
    # Obtener nuestros datos X e y. Los feautures y el traget
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis = 1)
        y = dataset[y]
        return X, y
    
    # Exportar el modelo de datos
    def model_export(self, clf,score):
        print(score)
        model_dir = os.path.abspath('./models')
        try:
            os.makedirs(model_dir, exist_ok=True)  # Crea la carpeta si no existe
            model_path = os.path.join(model_dir, 'best_model.pkl')
            joblib.dump(clf, model_path)
        except Exception as e:
            print(f"Error al exportar el modelo: {e}")
    