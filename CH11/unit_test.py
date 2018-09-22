#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/21/18
# Author: 😏 <smirk dot cao at gmail dot com>
from crf import *
from sympy import *
import argparse
import logging
import warnings
import unittest


class TestCRF(unittest.TestCase):
    def test_e111(self):
        # 针对11.1这个问题, 为什么要求非规范话的条件概率?
        # 参考下书中预测算法部分
        pass

    def test_e112(self):
        a01, a02, b11, b12, b21, b22, c11, c12, c21, c22 = symbols("a01, a02, b11, \
                                                                    b12, b21, b22, \
                                                                    c11, c12, c21, \
                                                                    c22")
        M1 = Matrix([[a01, a02],
                     [0,   0]])
        M2 = Matrix([[b11, b12],
                     [b21, b22]])

        M3 = Matrix([[c11, c12],
                     [c21, c22]])

        M4 = Matrix([[1, 0],
                     [1, 0]])
        Z = (M1 * M2 * M3 * M4)[0].expand()
        P = str(Z).replace(" ", "").split("+")
        logger.info(Z)
        logger.info(P)
        self.assertSetEqual(set(P),
                            {"a02*b21*c11", "a02*b21*c12", "a02*b22*c21", "a02*b22*c22",
                             "a01*b11*c11", "a01*b11*c12", "a01*b12*c21", "a01*b12*c22"})

    def test_e113(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    unittest.main()