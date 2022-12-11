import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset.head(5))

    X = dataset.drop(['country', 'score'], axis=1)
    Y = dataset['score']

    model = DecisionTreeRegressor()
    score = cross_val_score(
        model, X, Y, cv=5, scoring='neg_mean_squared_error')

    print(np.abs(np.mean(score)))
    print("-"*64)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    mse_values = []

    for train, test in kf.split(dataset):

        X_train = X.iloc[train]
        Y_train = Y.iloc[train]
        X_test = X.iloc[test]
        Y_test = Y.iloc[test]

        print(X_train, Y_train)

        model = DecisionTreeRegressor().fit(X_train, Y_train)
        predict = model.predict(X_test)
        mse_values.append(mean_squared_error(Y_test, predict))

    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))
