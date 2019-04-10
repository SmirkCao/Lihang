# -*-coding:utf-8-*-
# Project: CH06
# Filename: mnist_sample.py
# Author: 😏 <smirk dot cao at gmail dot com>

import pandas as pd
import numpy as np
import struct
import os

os.system("wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
os.system("wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
os.system("gunzip train-labels-idx1-ubyte.gz")
os.system("gunzip train-images-idx3-ubyte.gz")

with open("train-labels-idx1-ubyte", "br") as f:
    magic, num = struct.unpack('>II', f.read(8))
    labels = np.fromfile(f, dtype=np.uint8)

with open("train-images-idx3-ubyte", "br") as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows * cols)

df = pd.DataFrame(np.hstack((labels.reshape(-1, 1), images)),
                  columns=["label"] + ["pixel_%d" % i for i in range(images.shape[1])])
df.groupby("label").head(30).to_csv("train.csv", index=None)
os.system("mv train.csv ./input/.")
