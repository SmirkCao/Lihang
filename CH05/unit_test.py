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
        hd = cal_ent(raw_data[raw_data.columns[-1]])

        for idx in raw_data.columns[:-1]:
            hda = gain(raw_data[idx], raw_data[raw_data.columns[-1]])
            logger.info(hda)

        logger.info(hd)

    def test_e53(self):
        pass

    def test_e54(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()

