#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/21/18
# Author: 😏 <smirk dot cao at gmail dot com>
from crf import *
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
        pass

    def test_e113(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    unittest.main()