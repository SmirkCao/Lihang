# -*-coding:utf-8-*-
# Project: Lihang_CH5  
# Filename: decision_tree
# Author: Smirk <smirk dot cao at gmail dot com>
import pandas as pd
import numpy as np


class Tree(object):
    def __init__(self, eps, feas, name=None):
        self.tree_ = dict()
        self.feas = feas
        self.eps = eps
        if not name:
            self.name = "Decision Tree"
        else:
            self.name = name

    def fit(self, x, y):
        self.tree_ = self.build_tree(x, y, self.eps)
        return self.tree_

    def predict(self, x):
        label = str(x) + self.eps
        return label

    def build_tree(self, x, y, eps):
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
        max_gda = 0
        D = labels
        for fea in feas:
            A = x[:, fea]
            gda = gain(A, D)
            # uncomment this line for ex 5.3 gda result check
            # print(gda)
            if max_gda < gda:
                max_gda, max_fea = gda, fea

        # step4: eps
        if max_gda < eps:
            return max_label
        T = dict()
        sub_T = dict()
        for x_A in set(x[:, max_fea]):
            sub_D = D[x[:, max_fea] == x_A]
            sub_x = x[x[:, max_fea] == x_A, :]
            sub_x = np.delete(sub_x, max_fea, 1)
            # step6:
            sub_T[str(x_A) + "__" + str(sub_D.shape[0])] = self.build_tree(sub_x, sub_D, eps)
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
            if (type(tree[fea_idx]) == dict):
                tmp["name"] = fea_idx.split("__")[0]
                tmp["value"] = fea_idx.split("__")[1]
                tmp["children"] = self.describe_tree(tree[fea_idx])
            else:
                tmp["name"] = tree[fea_idx]
                tmp["value"] = 10
            rst.append(tmp)
        return rst

    def plot_tree(self, depth=3):
        from pyecharts import TreeMap

        data = self.describe_tree(self.tree_)
        treemap = TreeMap(self.name, "", width=800, height=500)
        treemap.use_theme("dark")
        treemap.add(self.name, data, is_label_show=True, label_pos='inside', treemap_left_depth=depth)
        return treemap


def cal_ent(x):
    """
    calculate shannon ent of x
    :param x:
    :return ent:
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
    df = pd.read_csv("./data/mdata_5-1.txt", index_col=0)

    # print(df.head())
    # print(cal_ent(df["类别"]))
    cols = df.columns
    # for col in cols[:-1]:
    #     print(col, "gain", gain(df[col], df[cols[-1]]))
    #     print(col, "gain_ratio", gain_ratio(df[col], df[cols[-1]]))
    X = df[cols[:-1]].values
    y = df[cols[-1]].values

    clf = Tree(eps=0.02, feas=cols)
    clf.fit(X, y)
    print(clf.tree_)
    clf.describe_tree(clf.tree_)

