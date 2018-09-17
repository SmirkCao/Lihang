#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: hmm
# Date: 9/17/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import warnings


class HMM(object):

    def __init__(self):
        self.A = None
        self.B = None
        self.p = None
        self.M = 0
        self.N = 0
        self.T = 0

    def _do_forward(self):
        pass

    def _do_backward(self):
        pass

    def _do_estep(self):
        pass

    def _do_mstep(self):
        pass

    def fit(self, X):
        # 估计模型参数
        return self

    def predict(self, X):
        rst = None
        return rst

    def sample(self):
        rst = None
        return rst

    def score(self):
        rst = None
        return rst


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

