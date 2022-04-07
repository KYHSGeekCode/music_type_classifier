import matplotlib.pyplot as plt
# https://github.com/scikit-learn/scikit-learn/issues/19137
# https://joyfuls.tistory.com/60
# https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
import numpy as np
import pandas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


def elbow():
    X = np.load("dataset_X.npy", allow_pickle=True)
    print(X.shape)  # (361, 6373)
    scaler = MinMaxScaler()
    sel = VarianceThreshold(threshold=(.97 * (1 - .97)))  #
    X = scaler.fit_transform(X)
    X = sel.fit_transform(X)
    print(X.shape)

    kmeans = KMeans()  # n_clusters=5
    visualizer = KElbowVisualizer(kmeans, k=(1, 10))
    visualizer.fit(X)
    visualizer.show()  # k = 3


def run():
    X = np.load("dataset_X.npy", allow_pickle=True)
    print(X.shape)  # (361, 6373)
    scaler = MinMaxScaler()
    sel = VarianceThreshold(threshold=(.97 * (1 - .97)))  #
    X = scaler.fit_transform(X)
    X = sel.fit_transform(X)
    print(X.shape)
    df = pd.DataFrame(X)
    df = df.reindex(df.var().sort_values().index, axis=1)
    kmeans = KMeans(n_clusters=3)  #
    kmeans.fit(df)

    # 결과 확인
    result_by_sklearn = df.copy()
    result_by_sklearn["cluster"] = kmeans.labels_
    print(result_by_sklearn.head())
    pandas.plotting.parallel_coordinates(result_by_sklearn, 'cluster', cols=df.columns[:15],
                                         color=(
                                             '#F900BF', '#0062FF', '#99FFF0', '#FF85B3', '#4ECDC4', '#C7F464',
                                             '#123456',
                                         ))
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_zlabel('Component 3', fontsize=15)
    ax.set_title('3 components', fontsize=20)
    targets = [0, 1, 2]
    print(targets)
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = result_by_sklearn['cluster'] == target
        ax.scatter(result_by_sklearn.loc[indicesToKeep, result_by_sklearn.columns[0]]
                   , result_by_sklearn.loc[indicesToKeep, result_by_sklearn.columns[1]]
                   , result_by_sklearn.loc[indicesToKeep, result_by_sklearn.columns[2]]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    elbow()
    run()
