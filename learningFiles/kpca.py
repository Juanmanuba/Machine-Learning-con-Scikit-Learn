import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

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

    kpca = KernelPCA(n_components=4, kernel='poly')
    # Para hacer que el PCA se ajuste a los datos que tenemos como X de entrenamiento
    kpca.fit(X_train)

    logistic = LogisticRegression(solver='lbfgs')

    # Aplicamos el PCA sobre el conjunto de entrenamiento
    dt_train = kpca.transform(X_train)
    # Aplicamos el PCA sobre el conjunto de prueba
    dt_test = kpca.transform(X_test)
    # Mandamos a nuestra regresion logistica los dos dataframes
    logistic.fit(dt_train, Y_train)

    print("Score KernelPCA:", logistic.score(
        dt_test, Y_test))  # Para el accuracy
