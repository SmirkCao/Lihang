#!/usr/bin/python
# coding=utf8
"""
1. 代码来源: https://vimsky.com/article/776.html
2. 代码中提到的公式:https://vimsky.com/article/714.html
3. 增加部分注释方便理解
4. Python3 语法
5. 回头看这代码路子挺清晰的, (sample_ep, Zx-> pyx->model_ep)->sigma
"""

from collections import defaultdict
import math


class MaxEnt:
    def __init__(self):
        self._samples = []              # 样本集, 元素是[y,x1,x2,...,xn]的元组
        self._Y = set([])               # 标签集合,相当于去重之后的y
        self._numXY = defaultdict(int)  # Key是(xi,yi)对，Value是count(xi,yi)
        self._N = 0                     # 样本数量，注意这里是满足要求的样本，并不对应了data中所有的数据，是过滤后的。
        self._n = 0                     # 特征对(xi,yi)总数量
        self._xyID = defaultdict(int)   # 对(x,y)对做的顺序编号(ID), Key是(xi,yi)对,Value是ID
        self._C = 0                     # 样本最大的特征数量,用于求参数时的迭代，见IIS原理说明
        self._ep_ = []                  # 样本分布的特征期望值
        self._ep = []                   # 模型分布的特征期望
        self._w = []                    # 对应n个特征的权值
        self._lastw = []                # 上一轮迭代的权值
        self._EPS = 0.01                # 判断是否收敛的阈值

    def load_data(self, filename):
        # 数据文件结构
        # [label]/t[play]/t[outlook]/t[temperature]/t[humidity]/t[windy]
        for line in open(filename, "r"):
            sample = line.strip().split("\t")
            # 注意这里说的特征是raw_data中的一个X特征向量中的每一个元素值, 对应一个特征取值， 参考后面X = sample[1:]
            # 至少: label + one feature
            if len(sample) < 2:
                continue
            y = sample[0]
            X = sample[1:]
            self._samples.append(sample)  # label + features
            self._Y.add(y)                # label : set
            # 注意, 数据是逐行读取的, 这里实际上做的是BOW的编码
            # 这里特征函数就是 f(x, y) in BOW
            for x in set(X):  # set给X去重, 对应了原始数据中， 同样的特征向量，会对应不同的label， 这种情况做计数。
                # 在整个样本集中统计TF，指定label的特征直方图。所以，在整个代码里面，特征就是不同的单词， numXY记录了词频
                # 在这个例子里面, 用(x, y)存储了特征的索引,而实际上, 如果用特征函数提取,
                self._numXY[(x, y)] += 1

    def _initparams(self):
        self._N = len(self._samples)
        self._n = len(self._numXY)  # 这个大小， 是BOW和标签组合的大小。
        self._C = max([len(sample) - 1 for sample in self._samples])  # -1 是为了去掉标签
        self._w = [0] * self._n
        self._lastw = self._w[:]
        self._sample_ep()

    def _convergence(self):
        for w, lw in zip(self._w, self._lastw):
            if math.fabs(w - lw) >= self._EPS:
                return False
        return True

    def _sample_ep(self):
        """
        样本期望, 特征函数f(x, y)关于经验分布\tilde p(x,y)的期望
        每个样本的直方图, 这个采用了稀疏存储, 实际上这里对应(m,n)的二维数组, 对应了不同的y情况下的特征直方图.
        :return:
        """
        self._ep_ = [0] * self._n
        # 计算方法参见公式(20)
        # 遍历特征求均值
        for i, xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy] * 1 / self._N
            self._xyID[xy] = i   # 在这里绑定了特征index和xy的关系

    def _zx(self, X):
        """
        因为后面利用最大熵模型计算条件概率分布pyx的时候,需要归一化, 所以求解Zx
        :param X:
        :return:
        """
        # calculate Z(X), 计算方法参见公式(15)
        ZX = 0
        for y in self._Y:
            s = 0
            for x in X:
                if (x, y) in self._numXY:
                    s += self._w[self._xyID[(x, y)]]
            # 如果x, y不存在, 那么掉不到这个if里面来, s 保持0, ZX变成计数
            ZX += math.exp(s)
        return ZX

    def _pyx(self, X):
        """
        注意这个里面X没有去重, 所以, 如果有个特征出现了两次, 会加两次.
        :param X:
        :return:
        """
        # 针对单一样本进行预测
        # calculate p(y|x), 计算方法参见公式(22)
        ZX = self._zx(X)
        results = []
        # Ck分类, 计算k次概率
        for y in self._Y:
            s = 0
            # 注意, 这里没有将X去重
            for x in X:
                # if (x, y) in self._numXY:  # 这个判断相当于指示函数的作用, 注意这里面f不是fi, 是f#
                #     s += self._w[self._xyID[(x, y)]]
                def f(x, y):
                    return 1 if (x, y) in self._numXY else 0
                # 注意这里, 如果xy的组合没有出现过, f(x,y) = 0, 这时w取得xyID=0时的值, f(x,y)相当于一个mask, 对w做了选择
                s += self._w[self._xyID[(x, y)]] * f(x, y)
            pyx = 1 / ZX * math.exp(s)
            results.append((y, pyx))
        return results

    def _model_ep(self):
        """
        注意这里, 考虑到不同的样本都在刷这个_ep,
        :return:
        """
        # 参见公式(21)
        self._ep = [0] * self._n
        for sample in self._samples:
            X = sample[1:]
            pyx = self._pyx(X)
            # 遍历特征, 针对出现过的特征做计算.
            for y, p in pyx:
                for x in X:
                    def f(x, y):
                        return 1 if (x, y) in self._numXY else 0

                    self._ep[self._xyID[(x, y)]] += p * 1 / self._N * f(x, y)
                    # if (x, y) in self._numXY:
                    #     self._ep[self._xyID[(x, y)]] += p * 1.0 / self._N

    def train(self, maxiter=1000):
        self._initparams()
        for i in range(0, maxiter):
            # print("Iter:%d..." % i)
            self._lastw = self._w[:]  # 保存上一轮权值
            self._model_ep()
            # 更新每个特征的权值
            for i, w in enumerate(self._w):
                # 参考公式(19)
                self._w[i] += 1.0 / self._C * math.log(self._ep_[i] / self._ep[i])
            # print(self._w)
            # 检查是否收敛
            if self._convergence():
                break

    def predict(self, X):
        # 这个实际上是predict_proba
        X_ = X.strip().split("\t")
        prob = self._pyx(X_)
        return prob


if __name__ == "__main__":
    maxent = MaxEnt()
    maxent.load_data('./Input/data.txt')
    maxent.train()
    print(maxent.predict("sunny\thot\thigh\tFALSE"))
    print(maxent.predict("overcast\thot\thigh\tFALSE"))
    print(maxent.predict("sunny\tcool\thigh\tTRUE"))
    print(maxent.predict("111\t1111\t2222\t1111"))
    print(maxent.predict("111\tcoll\thigh\t1111"))
    print(maxent.predict("high\thigh\thigh\t1111"))
    print(maxent.predict("FALSE\tFALSE\tFALSE\tFALSE"))
