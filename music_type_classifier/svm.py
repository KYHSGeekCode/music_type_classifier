# https://github.com/scikit-learn/scikit-learn/issues/19137
from sklearn.svm import SVC  # model 생성
from sklearn.model_selection import train_test_split  # train/test set
from sklearn.metrics import accuracy_score, confusion_matrix  # model 평가


def run():

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    print(x_train.shape)  # (105, 4)
    print(y_train.shape)  # (105,)

    # 3. model
    help(SVC)
    model = SVC()  # kernel = 'rbf'  : default
    # 데이터의 특징 때문에 분류정확도가 낮게 나오면 다른 커널 함수를 인수로 지정할 수 있음
    # 'linear', 'poly', 'rbf', 'sigmoid' 등

    model.fit(X=x_train, y=y_train)

    # 4. model 평가
    y_pred = model.predict(x_test)  # 예측치
    y_true = y_test  # 정답

    # 분류정확도 - accuracy_score 함수 사용해서 구하기
    acc = accuracy_score(y_true, y_pred)
    acc  # 0.9777777777777777

    con_mat = confusion_matrix(y_true, y_pred)
    con_mat
    '''
    array([[18,  0,  0],
           [ 0, 10,  0],
           [ 0,  1, 16]], dtype=int64)
    '''
    y_true[:10]  # [1, 2, 2, 1, 0, 2, 1, 0, 0, 1]
    y_pred[:10]  # [1, 2, 2, 1, 0, 1, 1, 0, 0, 1]

    type(con_mat)  # numpy.ndarray
    con_mat.shape  # (3, 3)

    # 분류정확도 - 식을 통해 구하기
    acc = (con_mat[0, 0] + con_mat[1, 1] + con_mat[2, 2]) / len(y_true)
    acc  # 0.9777777777777777

    ## iris 데이터의 경우, NB모델보다 SVM모델을 사용할 때 분류정확도가 더 올라간 것을 확인할 수 있음
