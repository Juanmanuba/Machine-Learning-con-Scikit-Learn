import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    # en cuantos grupos se dividio la data
    print("Etiquetas: ", meanshift.labels_)
    print("-"*64)

    print("Centros de cada grupo: ", meanshift.cluster_centers_)
    print("-"*64)

    print(meanshift.predict(X))
    print("-"*64)

    dataset['meanshift'] = meanshift.labels_

    print(dataset)
