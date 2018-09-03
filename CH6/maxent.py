# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: maxent
# Date: 8/24/18
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np
import math
import time
import argparse
import logging


class Maxent(object):
    def __init__(self, tol=1e-4, max_iter=100):
        self.X_ = None
        self.y_ = None
        self.n = None        # 特征数量
        self.N = None        # N 训练集样本容量
        self.M = None
        self.coef_ = None
        self.max_iter = max_iter
        self.tol = tol

    def build_dict(self):
        # 其实这个的做法, 是TFIDF嘛
        self.id2xy = dict()
        self.xy2id = dict()

        for idx, (x, y) in enumerate(self.Pxy):
            self.id2xy[idx] = (x, y)
            self.xy2id[(x, y)] = idx

    def _px_pxy(self, x, y):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        # 相当于按照特征统计了
        # 在这个例子里面, 相当于词表的大小是256, 也就是说特征就是灰度直方图
        for idx in range(len(x)):
            # 遍历每个样本
            x_, y_ = x[idx], y[idx]
            self.Y_.add(y_)
            # 统计样本中每个数据的px, pxy, 这个是为了求期望, 这里的每个数据, 实际上应该是经过特征提取之后的值.
            for x__ in x_:
                self.Pxy[(x__, y)] += 1     # 某个灰度值在对应的标签上的总数
                self.Px[x__] += 1           # 某个灰度值的总数

    def _EPxy(self):
        '''
        计算书中82页最下面那个期望
        这期望是特征函数f(x,y)关于经验分布的pxy期望值, 这里面做了简化, 针对训练样本所有的f(x,y)==1
        '''
        self.EPxy = defaultdict(float)
        # 针对特征函数提取期望, f(x, y)有n个
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def _pyx(self, x, y):
        result = 0
        for x_ in x:
            if self.fxy(x_, y):
                id = self.xy2id[(x_, y)]
                result += self.coef_[id]
        return math.exp(result), y

    def _pw(self, x):
        """
        计算书85页公式6.22和6.23, 这个表示的是最大熵模型.
        :param x:
        :return:
        """

        Pyxs = [(self._pyx(x, y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def _EPx(self):
        '''
        计算书83页最上面那个期望
        '''
        # self.EPx = [0.0 for i in range(self.n)]
        self.EPx = np.zeros(self.n)
        for i, X in enumerate(self.X_):
            Pyxs = self._pw(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        # 所以针对训练数据, f(x,y)是常数
        return (x, y) in self.xy2id

    def fit(self, x, y):
        """
        eq 6.34
        实际上这里是个熵差, plog(p)-plog(p)这种情况下, 对数差变成比值.

        :param x:
        :param y:
        :return: self: object
        """
        self.N = len(x)  # 训练集大小
        self.X_ = x
        self.y_ = set()

        self._px_pxy(x, y)

        self.n = len(self.Pxy)  # 书中(x,y)对数
        # 可认为是学习速率

        self.build_dict()
        self._EPxy()

        # IIS 算法流程 额, 也可能是GIS, 看下再
        # 初始化权重向全为0
        # self.w = [0.0 for i in range(self.n)]
        self.coef_ = np.zeros(self.n)
        # 整个这个过程都可以精简
        i = 0
        while i <= self.max_iter:
            logger.info('iterate times %d' % i)
            # sigmas = []
            self._EPx()
            self.M = 10000.0  # 书91页那个M，但实际操作中并没有用那个值

            # 拿到sigma向量
            # for i in range(self.n):
            #     sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
            #     sigmas.append(sigma)

            sigmas = 1/self.M*np.log(self.EPxy/self.EPx)
            # 好吧, 这份代码也是改的. 应该算法用的就是GIS了， 网上流传最广的应该就是这个GIS的例子了。
            # 文章中参考了这个文献《语言信息处理技术中的最大熵模型方法》以及另外一个博客文章
            # http://www.cnblogs.com/hexinuaa/p/3353479.html
            # 然而，这个文章没有介绍翻译的是什么， 源头是《A Brief Maxent Toturial》(Berger, 1995)
            # 这个里面应该是最原始的代码。https://vimsky.com/article/776.html
            # if len(filter(lambda x: abs(x) >= 0.01, sigmas)) == 0:
            #     break
            # 更新参数w
            self.coef_ = self.coef_ + sigmas
            # self.w = [self.w[i] + sigmas[i] for i in range(self.n)]
            i += 1
        return self

    def predict(self, x):
        """

        :param x:
        :return:
        """
        results = []
        for test in x:
            result = self._probality(test)
            results.append(max(result, key=lambda x: x[0])[1])
        return results

    def predict_proba(self, x):
        pass

def rebuild_features(features):
    '''
    将原feature的（a0,a1,a2,a3,a4,...）
    变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
    '''
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append(str(i) + '_' + str(f))
        new_features.append(new_feature)
    return new_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    logger.info('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels,
                                                                                test_size=0.33, random_state=23323)
    # 特征工程
    train_features = rebuild_features(train_features)
    test_features = rebuild_features(test_features)

    time_2 = time.time()
    logger.info('read data cost %d second' % (time_2 - time_1))
    logger.info('Start training')
    met = Maxent()
    met.fit(train_features, train_labels)

    time_3 = time.time()
    logger.info('training cost %d second' % (time_3 - time_2))
    logger.info('Start predicting')
    test_predict = met.predict(test_features)
    time_4 = time.time()
    logger.info('predicting cost %d second' % (time_4 - time_3))
    score = accuracy_score(test_labels, test_predict)
    logger.info("The accruacy socre is %d" % score)
