# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: maxent
# Date: 8/24/18
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np
import time
import logging


class Maxent(object):
    def __init__(self, tol=1e-4, max_iter=100):
        self.X_ = None
        self.y_ = None
        self.m = None        # 类别数量
        self.n = None        # 特征数量
        self.N = None        # N 训练集样本容量
        self.M = None
        self.coef_ = None
        self.label_names = defaultdict(int)
        self.feature_names = defaultdict(int)
        self.max_iter = max_iter
        self.tol = tol

    def _px_pxy(self, x, y):
        """
        统计TF, 这里面没有用稀疏存储的方式.
        :param x:
        :param y:
        :return:
        """
        self.Pxy = np.zeros((self.m, self.n))
        self.Px = np.zeros(self.n)

        # 相当于按照特征统计了
        # 在这个例子里面, 相当于词表的大小是256, 也就是说特征就是灰度直方图
        for idx in range(len(x)):
            # 遍历每个样本
            x_, y_ = x[idx], y[idx]
            # 某个灰度值在对应的标签上的总数
            for x__ in x_:
                self.Pxy[self.label_names[y_], self.feature_names[x__]] += 1
                self.Px[self.feature_names[x__]] += 1           # 某个灰度值的总数
        # 计算书中82页最下面那个期望
        # 这期望是特征函数f(x, y)
        # 关于经验分布的pxy期望值, 这里面做了简化, 针对训练样本所有的f(x, y) == 1
        self.EPxy = self.Pxy/self.N

    def _pw(self, x):
        """
        计算书85页公式6.22和6.23, 这个表示的是最大熵模型.
        :param x:
        :return:
        """
        mask = np.zeros(self.n)
        print("x->", type(x), x)
        for idx in x:
            mask[self.feature_names[idx]] = 1
        tmp = self.coef_*mask
        pw = np.exp(np.sum(tmp, axis=1))
        Z = np.sum(pw)
        return pw/Z

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
        self.N = len(x)  # 训练集大小
        self.X_ = x
        self.y_ = set(y)
        tmp = set(self.X_.flatten())
        self.feature_names = defaultdict(int, zip(tmp, range(len(tmp))))
        self.label_names = dict(zip(self.y_, range(len(self.y_))))
        self.n = len(self.feature_names)
        self.m = len(self.label_names)

        self._px_pxy(x, y)

        self.coef_ = np.zeros((self.m, self.n))
        # 整个这个过程都可以精简
        i = 0
        while i <= self.max_iter:
            logger.info('iterate times %d' % i)
            # sigmas = []
            self._EPx()
            self.M = 1000.0  # 书91页那个M，但实际操作中并没有用那个值
            sigmas = 1/self.M*np.log(self.EPxy/self.EPx)
            self.coef_ = self.coef_ + sigmas
            i += 1
        return self

    def predict(self, x):
        """

        :param x:
        :return:
        """
        rst = np.zeros(len(x))
        for idx, x_ in enumerate(x):
            tmp = self._pw(x_)
            print(tmp)
            rst[idx] = self.label_names[np.argmax(tmp)]
        return rst.astype(np.int64)

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to input image")
    # args = vars(ap.parse_args())

    logger.info('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('./Input/sub_train_binary.csv', sep=",", header=0)
    data = raw_data[:100].values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels,
                                                                                test_size=0.33, random_state=23323)

    logger.info("train test features %d, %d, %s" % (len(train_features), len(test_features), train_features[0]))
    time_2 = time.time()
    logger.info('read data cost %f second' % (time_2 - time_1))
    logger.info('Start training')
    met = Maxent(max_iter=100)
    print("train_features", train_features[:2])
    met.fit(train_features, train_labels)

    time_3 = time.time()
    logger.info('training cost %f second' % (time_3 - time_2))
    logger.info('Start predicting')
    test_predict = met.predict(test_features)
    time_4 = time.time()
    logger.info('predicting cost %d second' % (time_4 - time_3))
    score = accuracy_score(test_labels, test_predict)
    logger.info("The accruacy socre is %1.4f" % score)
    rst = met.predict_proba([np.zeros(len(train_features[0]))])
    logger.info(rst)
