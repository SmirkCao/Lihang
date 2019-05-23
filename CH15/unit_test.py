#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 5/23/19
# Author: 😏 <smirk dot cao at gmail dot com>

import unittest
import numpy as np


class TestSVDMethods(unittest.TestCase):

    def test_e_15_1(self):
        A = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 4],
                      [0, 3, 0, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, 0]])
                      
        u, s, vh = np.linalg.svd(A)
        ur, sr, vrh = np.linalg.svd(A, full_matrices=False)
        r = np.linalg.matrix_rank(A)
        print("\n")
        print(u)
        # 注意s和A.T*T
        # 奇异值，特征值的平方根
        print(s)
        # n阶对称实矩阵
        print(np.dot(A.T, A))
        print(vh)
        print(40*"*")
        print(ur)
        print(sr)
        print(vrh)
        print(40*"*")
        print(r)


if __name__ == "main":
    unittest.main()
