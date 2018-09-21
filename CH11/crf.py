#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: crf
# Date: 9/21/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import warnings


class CRF(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def decode(self):
        # 类似序列标注
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
