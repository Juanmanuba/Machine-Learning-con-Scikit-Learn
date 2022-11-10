import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset.head(5))

    X = dataset.drop(['country', 'score'], axis=1)
    Y = dataset[['score']]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)

    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, Y_train)
        predictions = estimador.predict(X_test)

        print('='*64)
        print(name)
        print('MSE: ', mean_squared_error(Y_test, predictions))

        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real ')
        plt.plot(Y_test, predictions, 'b')
        plt.show()
