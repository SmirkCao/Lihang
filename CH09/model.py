#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: model
# Date: 10/21/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging
import warnings


class TripleCoin(object):
    def __init__(self, pi=0, p=0, q=0):
        self.pi = pi
        self.p = p
        self.q = q

    def sample(self,
               n=10):
        """
        e9.1, 三硬币模型数据
        :param n:
        :return:
        """
        rst = np.empty(1)
        for n_iter in range(n):
            pi_ = np.random.binomial(1, self.pi, 1)
            if pi_:
                rst = np.hstack((rst, np.random.binomial(1, self.p, 1)))
            else:
                rst = np.hstack((rst, np.random.binomial(1, self.q, 1)))
        return rst[1:]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())