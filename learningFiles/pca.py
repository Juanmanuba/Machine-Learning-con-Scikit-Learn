import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/cancer.csv')

    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['LUNG_CANCER', 'GENDER'], axis=1)
    dt_target = dt_heart['LUNG_CANCER']

    dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, Y_train, Y_test = train_test_split(
        dt_features, dt_target, test_size=0.3, random_state=42)  # partimos los datos en el conjunto de prueba y conjunto de entrenamiento

    print(X_train.shape)  # (717, 13)

    print(X_test.shape)  # (717,)

    # n_components = min(n_muestras, n_features)
    # Llamada para aplicar el PCA y dejar solo 3 componentes, las mas relevantes
    pca = PCA(n_components=6)
    # Para hacer que el PCA se ajuste a los datos que tenemos como X de entrenamiento
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.stem(range(len(pca.explained_variance_)),
             pca.explained_variance_ratio_)  # En X nos muestra las componentes (Son 3: 0, 1, 2) y en Y nos muestra su porcentaje de importancia, su aporte de informacion basado en la varianza
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    # Aplicamos el PCA sobre el conjunto de entrenamiento
    dt_train = pca.transform(X_train)
    # Aplicamos el PCA sobre el conjunto de prueba
    dt_test = pca.transform(X_test)
    # Mandamos a nuestra regresion logistica los dos dataframes
    logistic.fit(dt_train, Y_train)

    print("Score PCA:", logistic.score(dt_test, Y_test))  # Para el accuracy

    # same for IPCA
    dt_train = ipca.transform(X_train)

    dt_test = ipca.transform(X_test)

    logistic.fit(dt_train, Y_train)

    print("Score IPCA:", logistic.score(dt_test, Y_test))  # Para el accuracy
