from collections import Counter

import matplotlib.pyplot as plt
# https://github.com/scikit-learn/scikit-learn/issues/19137
# https://joyfuls.tistory.com/60
# https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


def run():
    X = np.load("dataset_X.npy", allow_pickle=True)
    y = np.load("dataset_y.npy", allow_pickle=True)
    print(X.shape)  # (361, 6373)
    print(y.shape)
    y = np.array(list(map(str, y)))
    print(Counter(y))  # Counter({'4': 117, '2': 86, '3': 80, '1': 78})
    scaler = MinMaxScaler()
    sel = VarianceThreshold(threshold=(.85 * (1 - .85)))
    X = sel.fit_transform(X)
    X = scaler.fit_transform(X)
    print(X.shape)
    y = pd.DataFrame(y, columns=['label'])

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    finalDf = pd.concat([principalDf, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)
    targets = ['1', '2', '3', '4']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    run()
