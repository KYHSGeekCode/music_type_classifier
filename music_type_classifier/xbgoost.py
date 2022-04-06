from collections import Counter

# https://github.com/scikit-learn/scikit-learn/issues/19137
# https://joyfuls.tistory.com/60
# https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
# https://jonhyuk0922.tistory.com/114
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score  # model 평가
from sklearn.model_selection import train_test_split  # train/test set
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier  # xgboost 모델이 좋다길래 ..!!


def run():
    X = np.load("dataset_X.npy", allow_pickle=True)
    y = np.load("dataset_y.npy", allow_pickle=True)
    print(X.shape)
    print(y.shape)
    y = np.array([i - 1 for i in y])
    print(Counter(y))
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    print(x_train.shape)  # (252, 2668)
    print(y_train.shape)  # (361,)

    # 3. model
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, use_label_encoder=False)  # 1000개의 가지? epoch? , 0.05 학습률
    xgb.fit(x_train, y_train)  # 학습

    y_preds = xgb.predict(x_test)  # 검증

    print('Accuracy: %.2f' % accuracy_score(y_test, y_preds))  # Accuracy: 0.34


if __name__ == '__main__':
    run()
