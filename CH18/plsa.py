#! /usr/bin/env python

# -*- coding:utf-8 -*-
# Project:  Lihang
# Filename: plsa
# Date: 6/13/19
# Author: 😏 <smirk dot cao at gmail dot com>


class PLSA(object):
    def __init__(self, n_iter=100):
        self.w = None
        self.d = None
        self.z = None
        self.pz_wd = None
        self.pw_k = None
        self.pz_d = None
        self.n_iter = n_iter

    def fit(self, x):
        for idx in range(n_iter):
            self.e_step()
            self.m_step()
            if self.is_convergence():
                break

    def m_step(self,):
        self.pw_z = None
        self.pz_d = None

    def e_step(self,):
        self.pz_wd = None

    def is_convergence(self):
        pass