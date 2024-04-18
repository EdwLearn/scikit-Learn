from tools.utils import Utils
from tools.models import Model

if __name__ == '__main__':
    utils = Utils()
    models = Model()
    
    data = utils.load_from_csv('./data/felicidad.csv')
    X, y = utils.features_target(data,['score', 'rank', 'country'], ['score'])
    
    models.grid_training(X, y)