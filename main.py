from utils import Utils
from models import Models

if __name__ == '__main__':
    utils = Utils()
    models = Models()
    data = utils.load_from_csv('./data/cancer.csv')
    X, Y = utils. features_target(
        data, ['GENDER'], ['LUNG_CANCER'])

    models.grid_training(X, Y)

    print(data.head(5))
