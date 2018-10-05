#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 10/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
from dt import *
import pandas as pd
import numpy as np
import argparse
import logging
import unittest


class TestDT(unittest.TestCase):
    def test_e51(self):
        raw_data = pd.read_csv("./Input/data_5-1.txt")
        logger.info(raw_data)

    def test_e52(self):
        raw_data = pd.read_csv("./Input/data_5-1.txt")
        hd = dt._cal_entropy(raw_data[raw_data.columns[-1]])

        rst = np.zeros(raw_data.columns.shape[0] - 1)
        # note: _gain(ID, y) = ent(y)
        for idx, col in enumerate(raw_data.columns[1:-1]):
            hda = dt._gain(raw_data[col], raw_data[raw_data.columns[-1]])
            logger.info(hda)
            rst[idx] = hda
            # print(idx, col, hda)
        # logger.info(rst)
        # logger.info(np.argmax(rst))
        logger.info(hd)
        self.assertEqual(np.argmax(rst), 2) # index = 2 -> A3

    def test_e53(self):
        raw_data = pd.read_csv("./Input/data_5-1.txt")
        cols = raw_data.columns
        X = raw_data[cols[1:-1]]
        y = raw_data[cols[-1]]

        clf = dt()
        clf.fit(X, y)
        logger.info(clf.tree)
        rst = {'有自己的房子': {'否': {'有工作': {'否': {'否': None}, '是': {'是': None}}}, '是': {'是': None}}}
        self.assertEqual(rst, clf.tree)

    def test_e54(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()

