# -*-coding:utf-8-*-
# Project: CH6  
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>


from logistic_regression import *


def test_load_data(path_='./Input/train.csv'):
    X, y = load_data(path_)
    print(X.shape, y.shape)
    return X, y


def test_grident_decent():
    x, y = test_load_data()

    x = x[:100]
    y = y[:100]
    print(x.shape, y.shape)

    # R2 loss
    # def f(x_, y_, w_):
    #     # $L(w)=\frac{1}{2N}\sum_{i=1}^{N}(y-w\cdot x)^2$
    #     rst = 0
    #     m = y_.size
    #     rst = 1 / (2 * m) * np.sum((y_ - np.dot(x_, w_)) ** 2)
    #     return rst
    #
    # def g(x_, y_, w_):
    #     # $L'(w)=\frac{1}{2}x\cdot(w\cdot x-y)$
    #     # rst = []
    #     m = y_.size
    #     rst = 1 / m * np.dot(x_.T, (np.dot(x_, w_) - y_))
    #     return rst

    # LR loss
    # def f(x_, y_, w_):
    #     # Logistic Regression Loss
    #     m = y_.size
    #     rst_ = -(1 / m) * np.sum(np.dot(x_, w_) * y_ - np.log(1 + np.exp(np.dot(x_, w_))))
    #     return rst_
    #
    # def g(x_, y_, w_):
    #     m = y_.size
    #     rst_ = (1 / m) * np.dot(x_.T, y_ * sigmoid(np.dot(x_, w_)))
    #     return rst_
    #
    # def sigmoid(x_):
    #     p = np.exp(x_)
    #     p = p / (1 + p)
    #     return p

    clf = LogisticRegression()
    clf.f = f
    clf.g = g
    rst_w, rst_cols = clf.gradient_descent(x, y)

    # coef_.shape is (10, 784)
    time_1 = time.time()
    rst = np.array([rst_cols[idx] for idx in [np.argmax(rst) for rst in 1-sigmoid(np.dot(x, rst_w.T))]])
    rst = np.vstack([rst, y])
    time_2 = time.time()
    print('predict cost ', time_2 - time_1, ' second', '\n')
    print(rst.T)



def test_lr():
    x, y = test_load_data()

    x = x[:100]
    y = y[:100]
    print(x.shape, y.shape)

    clf = LogisticRegression()
    clf.f = f
    clf.g = g
    clf.fit(x, y)
    rst = clf.predict(x,)
    print(rst)


if __name__ == '__main__':
    # test_lr()
    # X, y = test_load_data(path_='./Input/train_10.csv')
    # test_grident_decent()
    test_lr()