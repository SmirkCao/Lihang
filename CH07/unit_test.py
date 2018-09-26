#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/27/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import unittest


class TestSVM(unittest.TestCase):
    def test_e71(self):
        # data 2.1

        pass

    def test_e72(self):
        # data 2.1
        pass

    def test_e72(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()
