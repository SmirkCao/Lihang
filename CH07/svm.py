#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: svm
# Date: 9/27/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import warnings


class SVM(object):
    def __init__(self):
        self.alpha = None

        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def predict_preba(self, X):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
 