#! /usr/bin/env python

# -*- coding:utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 6/13/19
# Author: 😏 <smirk dot cao at gmail dot com>
from plsa import PLSA
import matplotlib.pyplot as plt
import numpy as np


class TestPLSAMethods(unittest.TestCase):
    def test_plsa(self):
        w = None
        d = None
        z = None
        pz_wd = None
        pw_k = None
        pz_d = None
        plsa_test = PLSA()

