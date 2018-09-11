#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/11/18
# Author: 😏 <smirk dot cao at gmail dot com>
from hmm import *
import numpy as np
import pandas as pd
import logging
import warnings
import unittest


class TestMEMethods(unittest.TestCase):
    def test_e101(self):
        logger.info("Exercise 10.1")
        raw_data = pd.read_csv("./Input/data_10-1.txt", header=0, index_col=0)
        # print(raw_data)
        # print(list(raw_data.columns), list(raw_data.index))
        O = [0, 0, 1, 1, 0]
        # 以上为已知
        T= len(O)
        Q = set(raw_data.columns[-1-len(raw_data):-1])
        N = len(Q)
        V = set(raw_data.columns[:-1-len(raw_data)])
        M = len(V)
        A = raw_data[raw_data.columns[-1-len(raw_data):-1]].values
        B = raw_data[raw_data.columns[:-1 - len(raw_data)]].values
        B = B / np.sum(B, axis=1).reshape((-1, 1))
        B
        if raw_data[["pi"]].apply(np.isnan).values.flatten().sum() > 1:
            pi = [1/raw_data[["pi"]].apply(np.isnan).values.flatten().sum()]*N
        logger.info("\nT\n%s\nA\n%s\nB\n%s\npi\n%s\nM\n%s\nN\n%s\nO\n%s\nQ\n%s\nV\n%s"
                    % (T, A, B, pi, M, N, O, Q, V))
        pass

    def test_e102(self):
        logger.info("Exercise 10.2")
        raw_data = pd.read_csv("./Input/data_10-2.txt", header=0, index_col=0, na_values="None")
        O = [0, 1, 0]
        # 以上为已知
        T= len(O)
        Q = set(raw_data.columns[-1-len(raw_data):-1])
        N = len(Q)
        V = set(raw_data.columns[:-1-len(raw_data)])
        M = len(V)
        A = raw_data[raw_data.columns[-1-len(raw_data):-1]].values
        B = raw_data[raw_data.columns[:-1 - len(raw_data)]].values
        B = B / np.sum(B, axis=1).reshape((-1, 1))
        B
        if raw_data[["pi"]].apply(np.isnan).values.flatten().sum() > 1:
            pi = [raw_data[["pi"]].apply(np.isnan).values.flatten().sum()]*N
        else:
            pi = raw_data[["pi"]].values.flatten()
        logger.info("\nT\n%s\nA\n%s\nB\n%s\npi\n%s\nM\n%s\nN\n%s\nO\n%s\nQ\n%s\nV\n%s"
                    % (T, A, B, pi, M, N, O, Q, V))
        # forward
        logger.info(pi*B[..., O[0]])
        logger.info(np.dot(pi*B[..., O[0]], A)*B[..., O[1]])
        logger.info(np.dot(np.dot(pi*B[..., O[0]], A)*B[..., O[1]], A)*B[..., O[2]])
        logger.info(np.sum(np.dot(np.dot(pi*B[..., O[0]], A)*B[..., O[1]], A)*B[..., O[2]]))

    def test_e103(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    unittest.main()
