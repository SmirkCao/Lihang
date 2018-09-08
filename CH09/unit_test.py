#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/6/18
# Author: 😏 <smirk dot cao at gmail dot com>
from gmm import *
import logging
import unittest


class TestMEMethods(unittest.TestCase):
    def test_e91(self):
        # 这个是个伯努利分布, 例子讲的是EM算法, 不是GMM. 理解这里的关系
        X = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
        gmm = GMM()
        gmm.fit(X)
        rst = gmm.predict(X)
        print(rst)

    def test_t93(self):
        pass

    def test_t91(self):
        pass

    def test_simulation(self):
        # 使用高斯生成数据, 然后用gmm拿到模型参数, 对比生成参数与学习到的参数.
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    unittest.main()
