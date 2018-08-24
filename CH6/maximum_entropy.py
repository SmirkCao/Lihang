# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: maximum_entropy
# Date: 8/24/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())