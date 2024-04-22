# Importa las clases Utils y Model de la carpeta tools
from tools.utils import Utils
from tools.models import Model

# Define la función principal
if __name__ == '__main__':
    # Crea un objeto de la clase Utils
    utils = Utils()
    # Crea un objeto de la clase Model
    models = Model()

    # Carga los datos del archivo csv 'felicidad.csv' y los guarda en la variable data
    data = utils.load_from_csv('./data/felicidad.csv')
    # Separa las variables independientes (X) y la variable dependiente (y)
    X, y = utils.features_target(data,['score', 'rank', 'country'], ['score'])

    # Realiza la búsqueda en malla y entrena los modelos con los datos
    models.grid_training(X, y)
