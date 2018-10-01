# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: nb
# Date: 8/16/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import pandas as pd
import argparse
import logging


class NB(object):

    def __init__(self,
                 lambda_):
        self.lambda_ = lambda_
        self.classes_ = None
        self.prior_ = None
        self.class_prior_ = None
        self.class_count_ = None

    def fit(self, x_, y_):
        self.classes_ = list(set(y_))
        # to df
        x_ = pd.DataFrame(x_)
        y_ = pd.DataFrame(y_)

        self.class_count_ = y_[y_.columns[0]].value_counts()
        self.class_prior_ = self.class_count_/y_.shape[0]
        # prior
        self.prior_ = dict()
        for idx in x_.columns:
            for j in self.classes_:
                p_x_y = x_[(y_ == j).values][idx].value_counts()
                for i in p_x_y.index:
                    self.prior_[(idx, i, j)] = p_x_y[i]/self.class_count_[j]

    def predict(self, x_):
        rst = []
        for class_ in self.classes_:
            py = self.class_prior_[class_]
            pxy = 1
            for idx, x in enumerate(x_):
                pxy *= self.prior_[(idx, x, class_)]

            rst.append(py*pxy)
        return self.classes_[np.argmax(rst)]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
