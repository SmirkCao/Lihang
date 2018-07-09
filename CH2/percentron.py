# -*-coding:utf-8-*-
# Project: CH2  
# Filename: percentron
# Author: 😏 <smirk dot cao at gmail dot com>
import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self,
                 max_iter=5000,
                 eta=0.00001):
        self.eta_ = eta
        self.max_iter_ = max_iter

    def fit(self, x_, y_):
        self.w = np.zeros(x_[0].shape[0] + 1)
        correct_count = 0
        n_iter_ = 0

        while n_iter_ < self.max_iter_:
            index = random.randint(0, y_.shape[0] - 1)
            xx_ = np.hstack([x_[index], 1])
            yy_ = 2 * y_[index] - 1
            wx = sum((self.w*xx_).T)

            if wx * yy_ > 0:
                correct_count += 1
                if correct_count > self.max_iter_:
                    break
                continue

            self.w += self.eta_*yy_*xx_
            n_iter_ += 1

    def predict(self, x_):
        x_ = np.hstack([x_, np.ones(x_.shape[0]).reshape((-1, 1))])
        rst = np.array([1 if rst else 0 for rst in sum((x_ * self.w).T) > 0])
        return rst


if __name__ == '__main__':
    print('Start read data')
    raw_data = pd.read_csv('./data/train_binary.csv', header=0)
    data = raw_data.values

    X = data[0::, 1::]
    y = data[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)

    print('Start training')
    p = Perceptron()
    p.fit(X_train, y_train)

    print('Start predicting')
    test_predict = p.predict(X_test)

    score = accuracy_score(y_test, test_predict)
    print("The accruacy socre is ", score)
