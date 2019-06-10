#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: mcmc
# Date: 6/10/19
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np


class Sampler(object):
    def __init__(self, px=None, qx=None, c=1):
        self.px = px
        self.qx = qx
        self.c = c

    def sample(self, n_samples):
        pass


class Rejection(Sampler):
    # algorithm 19.1
    def sample(self, n_samples):
        # step 1: assign px, qx
        assert self.px is not None
        assert self.qx is not None
        print("\n", n_samples, "samples")
        # step 2:
        c = self.c
        count = 0
        rst = []
        for idx in range(0, n_samples):
            while True:
                x_star = np.random.normal(0.76, 1, 1)[0]
                u = np.random.uniform(0, 1)
                # step 3:
                if u <= self.px(x_star)/c/self.qx(x_star):
                    rst.append(x_star)
                    break
                else:
                    count += 1
        assert len(rst) == n_samples
        # print("Reject count: ", count)
        return rst, count


class MetropolisHastings(Sampler):
    # algorithm 19.2
    def sample(self, n_samples, k):
        print("k", k)
        print("samples", n_samples)


class Gibbs(Sampler):
    # algorithm 19.3
    def sample(self):
        pass
