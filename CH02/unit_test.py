# -*-coding:utf-8-*-
# Project: CH2  
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>
from perceptron import *
import numpy as np


def test_logic(x_, y_):
    p = Perceptron(max_iter=100, eta=0.01)
    p.fit(x_, y_)
    print("w,b", p.w)
    print(p.predict(x_))


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y_and = np.array([1, 0, 0, 0])
    y_or = np.array([1, 1, 1, 0])
    y_not = np.array([0, 0, 1, 1])
    """
    学习率大分不开
    and
    w,b [ 0.02  0.01  0.03 -0.05]
    [1 0 0 0 0 0 0 0]
    or
    w,b [ 0.02  0.02  0.02 -0.01]
    [1 1 1 1 1 0 1 1]
    not
    w,b [-0.03  0.01  0.    0.01]
    [0 0 0 1 1 1 0 1]
    """
    X = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0],
                  [0, 1, 1], [0, 1, 0],
                  [0, 0, 0], [1, 0, 1], [0, 0, 1]])
    y_and = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    y_or = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    y_not = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    """
    and
    w,b [ 3.  3.  1. -4.]
    [1 1 0 0 0 0 0 0]
    or
    w,b [ 2.  2.  2. -1.]
    [1 1 1 1 1 0 1 1]
    not
    w,b [-4.  1.  0.  1.]
    [0 0 0 1 1 1 0 1]
    """
    print("and")
    test_logic(X, y_and)
    print("or")
    test_logic(X, y_or)
    print("not")
    test_logic(X, y_not)

