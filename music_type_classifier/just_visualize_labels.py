from collections import Counter

import matplotlib.pyplot as plt
# https://github.com/scikit-learn/scikit-learn/issues/19137
# https://joyfuls.tistory.com/60
# https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
import numpy as np
import pandas as pd
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
    sel = VarianceThreshold(threshold=(.97 * (1 - .97)))
    X = scaler.fit_transform(X)
    X = sel.fit_transform(X)
    print(X.shape)
    y = pd.DataFrame(y, columns=['label'])
    df = pd.DataFrame(X)
    df = df.reindex(df.var().sort_values().index, axis=1)

    finalDf = pd.concat([df, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_zlabel('Component 3', fontsize=15)
    ax.set_title('3 components by variance', fontsize=20)
    targets = ['1', '2', '3', '4']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, df.columns[0]]
                   , finalDf.loc[indicesToKeep, df.columns[1]]
                   , finalDf.loc[indicesToKeep, df.columns[2]]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    run()
