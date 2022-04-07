# https://github.com/scikit-learn/scikit-learn/issues/19137
# https://joyfuls.tistory.com/60
# https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC  # model 생성
from sklearn.model_selection import train_test_split  # train/test set
from sklearn.metrics import accuracy_score, confusion_matrix  # model 평가
from collections import Counter
from sklearn.feature_selection import VarianceThreshold


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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    print(x_train.shape)  # (252, 1820)
    print(y_train.shape)  # (252,)

    # 3. model
    model = SVC(kernel='linear')  # kernel = 'rbf'  : default
    # 데이터의 특징 때문에 분류정확도가 낮게 나오면 다른 커널 함수를 인수로 지정할 수 있음
    # 'linear' is too slow; others are bad
    # 'linear', 'poly', 'rbf', 'sigmoid' 등

    model.fit(X=x_train, y=y_train)

    # 4. model 평가
    y_pred = model.predict(x_test)  # 예측치
    y_true = y_test  # 정답

    # 분류정확도 - accuracy_score 함수 사용해서 구하기
    acc = accuracy_score(y_true, y_pred)
    print(acc)  # 0.41284403669724773

    con_mat = confusion_matrix(y_true, y_pred)
    print(con_mat)
    '''
    [[ 2  4  0 17]
    [ 1  9  1 20]
    [ 1  4  0 23]
    [ 2  3  0 22]]
    '''
    print(y_true[:10])  # [3 3 2 3 3 2 2 2 4 4]
    print(y_pred[:10])  # [4 4 4 4 4 4 4 4 4 4]

    print(type(con_mat))  # numpy.ndarray
    print(con_mat.shape)  # (4, 4)

    # 분류정확도 - 식을 통해 구하기
    acc = (con_mat[0, 0] + con_mat[1, 1] + con_mat[2, 2] + con_mat[3, 3]) / len(y_true)
    print(acc)  # 0.44036697247706424


if __name__ == '__main__':
    run()
