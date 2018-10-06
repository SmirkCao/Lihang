#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: dt
# Date: 10/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Tree(object):

    def __init__(self,
                 eps,
                 feas,
                 name=None,
                 criterion="entropy"):

        self.tree_ = dict()
        self.feas = feas
        self.eps = eps
        self.criterion = criterion
        if not name:
            self.name = "Decision Tree"
        else:
            self.name = name

    def fit(self, x, y):
        self.tree_ = self._build_tree(x, y, self.eps)
        return self.tree_

    def predict(self, x, x_tree=None):

        if len(x.shape) == 2:
            rst = []
            for x_ in x:
                rst.append(self.predict(x_))
            return rst

        if not x_tree:
            x_tree = self.tree_
        tree_key = list(x_tree.keys())[0]
        x_fea = tree_key.split("__")[0]
        x_idx = clf.feas.index(x_fea)
        x_tree = x_tree[tree_key]
        for key in x_tree.keys():
            if key.split("__")[0] == x[x_idx]:
                tree_key = key
                x_tree = x_tree[tree_key]
        if type(x_tree) == dict:
            return self.predict(x, x_tree)
        else:
            return x_tree

    def _build_tree(self, x, y, eps):
        feas = np.arange(x.shape[1])
        labels = y
        # step1: same label
        if len(set(labels)) == 1:
            return labels[0]

        max_label = max([(i, len(list(filter(lambda tmp: tmp == i, labels)))) for i in set(labels)]
                        , key=lambda tmp: tmp[1])[0]

        # step2: empty features
        if len(feas) == 0:
            return max_label

        # step3: feature selection
        max_fea = 0
        max_criterion = 0
        D = labels
        for fea in feas:
            A = x[:, fea]
            if self.criterion == "entropy":
                gda = gain(A, D)
            elif self.criterion == "gr":
                gda = gain_ratio(A, D)
            elif self.criterion == "gini":
                pass
            # uncomment this line for ex 5.3 gda result check
            # print(gda)
            if max_criterion < gda:
                max_criterion, max_fea = gda, fea

        # step4: eps
        if max_criterion < eps:
            return max_label
        T = dict()
        sub_T = dict()
        for x_A in set(x[:, max_fea]):
            sub_D = D[x[:, max_fea] == x_A]
            sub_x = x[x[:, max_fea] == x_A, :]
            sub_x = np.delete(sub_x, max_fea, 1)
            # step6:
            sub_T[str(x_A) + "__" + str(sub_D.shape[0])] = self._build_tree(sub_x, sub_D, eps)
        # step5: T
        # self.tree_[max_fea] = sub_tree
        T[str(self.feas[max_fea]) + "__" + str(D.shape[0])] = sub_T
        return T

    def describe_tree(self, tree=None):
        rst = []
        if not tree:
            tree = self.tree_
        for fea_idx in tree.keys():
            tmp = dict()
            tmp["name"] = fea_idx.split("__")[0]
            tmp["value"] = fea_idx.split("__")[1]
            if type(tree[fea_idx]) == dict:
                tmp["children"] = self.describe_tree(tree[fea_idx])
            else:
                tmp["children"] = [{"name": tree[fea_idx], "value": 10}]
            rst.append(tmp)
        return rst

    def plot_tree(self, depth=3):
        from pyecharts import TreeMap

        data = self.describe_tree(self.tree_)
        tree_map = TreeMap(self.name, "", width=800, height=500)
        tree_map.use_theme("dark")
        tree_map.add(self.name, data, is_label_show=True, label_pos='inside', treemap_left_depth=depth)
        return tree_map

    def _choose_best_fea(self, x_, y_):
        rst = []
        for i in range(x_.shape[1]):
            for s_value in set(x_[:, i]):
                rst.append((i, s_value, gini(x_[:, i], y_=y_, s_=s_value)))
        rst.sort(key=lambda x: x[2])
        return rst[0] if rst != [] else rst

    def _build_cart(self, x_, y_):
        """

        :param x_:
        :param y_:
        :return: cart_tree
        """
        cart = dict()
        fea = self._choose_best_fea(x_, y_)
        if len(fea) > 0:
            key = fea[:2]
            idx_l = x_[:, fea[0]] == fea[1]
            idx_r = x_[:, fea[0]] != fea[1]
            if len(set(y_)) == 1:
                cart = y_[0]
            elif fea[2] == 0:
                cart[key] = {"left": y_[idx_l][0], "right": y_[idx_r][0]}
            else:

                if x_.shape[1] > 1:
                    x_l = x_[idx_l, :]
                    x_r = x_[idx_r, :]
                else:
                    x_l = x_[idx_l]
                    x_r = x_[idx_r]
                y_l = y_[idx_l]
                y_r = y_[idx_r]
                cart[key] = {"left": self._build_cart(x_l, y_l),
                             "right": self._build_cart(x_r, y_r)}
        else:
            pass
        return cart

    def create_cart(self, x_, y_):
        return self._build_cart(x_, y_)

    def pruning(self):
        # TODO: Pruning
        pass


def gini(x_, y_=None, s_=None):
    """

    :param x_: Feature A
    :param y_: Class D
    :param s_: split threshold
    :return: gini(y,x) or gini(y)
    """
    if y_ is None:
        x_values = list(set(x_))
        p = 0
        for x_value in x_values:
            p += (x_[x_ == x_value].shape[0] / x_.shape[0]) ** 2
        return 1 - p
    else:
        D1 = y_[x_ == s_]
        D2 = y_[x_ != s_]
        rst = D1.shape[0]/y_.shape[0]*gini(D1)+D2.shape[0]/y_.shape[0]*gini(D2)
        return rst


def cal_ent(x):
    """
    calculate shannon ent of x
    :param x:
    :return ent: H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}
    """
    x_values = list(set(x))
    ent = 0
    for x_value in x_values:
        p = x[x == x_value].shape[0]/x.shape[0]
        ent -= p*np.log2(p)
    return ent


def cal_condition_ent(x, y):
    """
    calculate condition ent(y|x)
    :param x: feature
    :param y: class
    :return: ent(y|x)
    """
    ent = 0
    x_values = set(x)
    for x_value in x_values:
        sub_y = y[x == x_value]
        tmp_ent = cal_ent(sub_y)
        p = sub_y.shape[0]/y.shape[0]
        ent += p*tmp_ent
    return ent


def gain(x, y):
    """
    calculate information g(y, x)
    :param x: feature
    :param y: class
    :return: gain
    """
    return cal_ent(y) - cal_condition_ent(x, y)


def gain_ratio(x, y):
    """
    calculate gain ration gr(y, x)
    :param x: feature
    :param y: class
    :return: gr
    """
    return gain(x, y)/cal_ent(x)


if __name__ == '__main__':
    df = pd.read_csv("./Input/mdata_5-1.txt", index_col=0)

    # print(df.head())
    # print(cal_ent(df["类别"]))
    cols = df.columns.tolist()
    # for col in cols[:-1]:
    #     print(col, "gain", gain(df[col], df[cols[-1]]))
    #     print(col, "gain_ratio", gain_ratio(df[col], df[cols[-1]]))
    X = df[cols[:-1]].values
    y = df[cols[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Tree(eps=0.02, feas=cols, criterion="gr")
    clf.fit(X_train, y_train)
    # test ID3 fit predict
    print(clf.tree_)
    clf.describe_tree(clf.tree_)

    print(clf.predict(X_test))
    # for x_test in X_test:
    #     print(clf.predict(x_test))

    # test cart build EX5.4
    # cart_tree = clf.create_cart(X, y)
    # print(cart_tree)


