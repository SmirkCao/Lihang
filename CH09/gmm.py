#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: gmm
# Date: 9/5/18
# Author: 😏 <smirk dot cao at gmail dot com>

import numpy as np
import argparse
import logging

tol = 0.0001


def get_dummy():
    mu1 = 5
    mu2 = 6
    sigma1 = 0.1
    sigma2 = 0.5
    alpha1 = 0.4
    alpha2 = 0.6

    N = 4000
    X = np.hstack([np.random.normal(mu1, sigma1, int(alpha1*N)), np.random.normal(mu2, sigma2, int(alpha2*N))])
    return np.mat(X)


def gmm(X):
    """
    todo: 封装, 输入检测, 使用矩阵操作要比循环快很多, 一两个数量级的差异, 可以做个对比
    :param X:
    :return:
    """
    k = 2
    N = X.shape[1]
    mu_ = np.random.rand(k, 1)
    sigma_ = np.random.rand(k, 1)
    alpha_ = np.random.rand(k, 1)
    logger.info('\n init mu= \n%s \n init sigma=\n%s \n init alpha=\n%s' % (mu_, sigma_, alpha_))

    X_ = np.reshape(np.tile(X, 2), (-1, 2), order="F")
    for n_iter in range(1000):
        # numerator_ = np.exp(-1.0 * np.power((X_ - mu_.T), 2) / (np.sqrt(2.0 * np.pi) * sigma_.T))
        # 迭代过程中, 常数的计算不是特别重要, 这里去掉之后更容易收敛
        numerator_ = np.exp(-1.0 * np.power((X_ - mu_.T), 2) / sigma_.T)
        numerator_ = np.multiply(numerator_, alpha_.T)
        dominator_ = np.sum(numerator_, axis=1)
        # \hat\gamma_{jk}
        posterior_ = numerator_/dominator_

        mu_last = mu_
        alpha_last = alpha_
        sigma_last = sigma_

        Z = np.sum(posterior_, axis=0).T
        alpha_ = Z/N
        sigma_ = np.sqrt(np.sum(np.multiply(posterior_, np.power((X_ - mu_.T), 2)), axis=0)/Z.T).T
        mu_ = (np.sum(np.multiply(posterior_, X_), axis=0)/Z.T).T
        if ((abs(mu_ - mu_last)).sum() < tol) and \
                ((abs(alpha_ - alpha_last)).sum() < tol) and \
                ((abs(sigma_ - sigma_last)).sum() < tol):
            logger.info('\n mu= \n%s \n sigma=\n%s \n alpha=\n%s' % (mu_, sigma_, alpha_))
            logger.info(n_iter)
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    X = get_dummy()
    gmm(X)
