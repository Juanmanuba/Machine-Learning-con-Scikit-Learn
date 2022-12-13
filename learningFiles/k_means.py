import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv('./data/cancer.csv')
    print(dataset.head(10))

    X = dataset.drop(['LUNG_CANCER', 'GENDER'], axis=1)

    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=8).fit(X)
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("-"*64)

    print(kmeans.predict(X))
    print("-"*64)

    dataset['group'] = kmeans.predict(X)

    dataset.to_csv('file_name.csv')
    print(dataset)
