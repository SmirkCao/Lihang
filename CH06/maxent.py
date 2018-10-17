# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: maxent
# Date: 8/24/18
# Author: 😏 <smirk dot cao at gmail dot com>

from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import logging


class Maxent(object):
    """
    注意这里面最大的特征的数量是mxn, 但是实际上这个mxn的矩阵会非常稀疏.
    """
    def __init__(self, tol=1e-4, max_iter=100):
        self.X_ = None
        self.y_ = None
        self.m = None        # 类别数量
        self.n = None        # 词表的数量
        self.N = None        # N 训练集样本容量
        self.M = None
        self.coef_ = None
        self.label_names = defaultdict(int)
        self.feature_names = defaultdict(int)
        self.max_iter = max_iter
        self.tol = tol

    def _px_pxy(self, x, y):
        """
        统计TF, 这里面没有用稀疏存储的方式. 所以这里会有很多的0, 包括后面的E也会有很多零, 需要处理掉除零的问题.
        这里x, y是全量的数据,
        :param x:
        :param y:
        :return:
        """
        self.Pxy = np.zeros((self.m, self.n))
        self.Px = np.zeros(self.n)

        # 相当于按照特征统计了
        # 在这个例子里面, 相当于词表的大小是256, 对应的特征就是词表和类别组合

        for x_, y_ in zip(x, y):
            # 遍历每个样本, 某个灰度值在对应的标签上的总数, 注意每个样本中, 某个x__出现多少次的贡献认为都一样
            for x__ in set(x_):
                self.Pxy[self.label_names[y_], self.feature_names[x__]] += 1
                self.Px[self.feature_names[x__]] += 1           # 某个灰度值的总数
        # 计算书中82页最下面那个期望
        # 这期望是特征函数f(x, y)
        # 关于经验分布的pxy期望值, 这里面做了简化, 针对训练样本所有的f(x, y) == 1
        self.EPxy = self.Pxy/self.N

    def _pw(self, x):
        """
        计算书85页公式6.22和6.23, 这个表示的是最大熵模型.
        mask相当于给
        :param x:
        :return:
        """
        mask = np.zeros(self.n+1)
        # print("x->", type(x), x)
        for idx in x:
            mask[self.feature_names[idx]] = 1
        tmp = self.coef_*mask[1:]
        pw = np.exp(np.sum(tmp, axis=1))
        Z = np.sum(pw)
        pw = pw/Z
        return pw

    def _EPx(self):
        """
        计算书83页最上面那个期望
        对于同样的y, Ex是一样的, 所以这个矩阵其实用长度是n的向量表示就可以了.
        :return:
        """
        self.EPx = np.zeros((self.m, self.n))
        for X in self.X_:
            pw = self._pw(X)
            pw = pw.reshape(self.m, 1)
            px = self.Px.reshape(1, self.n)
            self.EPx += pw*px / self.N

    def fit(self, x, y):
        """
        eq 6.34
        实际上这里是个熵差, plog(p)-plog(p)这种情况下, 对数差变成比值.

        :param x:
        :param y:
        :return: self: object
        """
        self.X_ = x
        self.y_ = list(set(y))
        tmp = set(self.X_.flatten())
        self.feature_names = defaultdict(int, zip(tmp, range(1, len(tmp)+1)))   # 从1开始编码
        self.label_names = dict(zip(self.y_, range(len(self.y_))))
        self.n = len(self.feature_names)+1  # for default 0
        self.m = len(self.label_names)
        self.N = len(x)  # 训练集大小

        self._px_pxy(x, y)

        self.coef_ = np.zeros((self.m, self.n))
        # 整个这个过程都可以精简
        i = 0
        while i <= self.max_iter:
            logger.info('iterate times %d' % i)
            # sigmas = []
            self._EPx()
            self.M = 1000  # 书91页那个M，但实际操作中并没有用那个值
            # TODO: 理解f^\#
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp = np.true_divide(self.EPxy, self.EPx)
                tmp[tmp == np.inf] = 0
                tmp = np.nan_to_num(tmp)
            sigmas = np.where(tmp != 0, 1/self.M*np.log(tmp), 0)  # TODO: 还有除零的异常, 有空再看下
            self.coef_ = self.coef_ + sigmas
            i += 1
        return self

    def predict(self, x):
        """

        :param x:
        :return:
        """
        rst = np.zeros(len(x), dtype=np.int64)
        for idx, x_ in enumerate(x):
            tmp = self._pw(x_)
            print(tmp, np.argmax(tmp), self.label_names)
            rst[idx] = self.label_names[self.y_[np.argmax(tmp)]]
        return np.array([self.y_[idx] for idx in rst])

    def predict_proba(self, x):
        """

        :param x:
        :return:
        """
        rst = []
        for idx, x_ in enumerate(x):
            tmp = self._pw(x_)
            rst.append(tmp)
        return rst


def load_data(path=None):
    if path is None:
        from sklearn.datasets import load_digits
        raw_data = load_digits()
        imgs, labels = raw_data.data, raw_data.target
    else:
        raw_data = pd.read_csv(path, sep="[,\t]", header=0, engine="python")
        data = raw_data.values
        imgs, labels = data[0::, 1::], data[::, 0]
    return imgs, labels


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())
