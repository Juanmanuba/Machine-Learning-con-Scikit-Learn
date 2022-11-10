import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset[['gdp', 'family', 'lifexp', 'freedom',
                 'corruption', 'generosity', 'dystopia']]  # decidimos cuales features incluir, doble corchete para decirle a pd que estamos operando sobre las columnas
    Y = dataset[['score']]

    print(X.shape)
    print(Y.shape)

    # partimos los datos en el conjunto de prueba y conjunto de entrenamiento
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    modelLinear = LinearRegression().fit(X_train, Y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, Y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=0.02).fit(X_train, Y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    modelElasticNet = ElasticNet(alpha=1, l1_ratio=0.8, random_state=0)
    modelElasticNet.fit(X_train, Y_train)
    y_predict_elasticNet = modelElasticNet.predict(X_test)

    linear_loss = mean_squared_error(Y_test, y_predict_linear)
    print("Linear loss: ", linear_loss)

    lasso_loss = mean_squared_error(Y_test, y_predict_lasso)
    print("Lasso loss: ", lasso_loss)

    ridge_loss = mean_squared_error(Y_test, y_predict_ridge)
    print("Ridge loss: ", ridge_loss)

    elasticNet_loss = mean_squared_error(Y_test, y_predict_elasticNet)
    print("ElasticNet loss: ", elasticNet_loss)

    print("-"*64)
    print("Coeficientes Lasso:")
    print(modelLasso.coef_)

    print("-"*64)
    print("Coeficientes Ridge:")
    print(modelRidge.coef_)

    print("-"*64)
    print("Coeficientes ElasticNet:")
    print(modelElasticNet.coef_)
