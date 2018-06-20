# -*-coding:utf-8-*-
# Project: Lihang_CH5  
# Filename: decision_tree
# Author: Smirk <smirk dot cao at gmail dot com>
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Tree(object):
    # TODO: CART build_tree
    # TODO: CART pruning

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
            if (type(tree[fea_idx])==dict):
                tmp["children"] = self.describe_tree(tree[fea_idx])
            else:
                tmp["children"] = [{"name": tree[fea_idx], "value": 10}]
            rst.append(tmp)
        return rst

    def plot_tree(self, depth=3):
        from pyecharts import TreeMap

        data = self.describe_tree(self.tree_)
        treemap = TreeMap(self.name, "", width=800, height=500)
        treemap.use_theme("dark")
        treemap.add(self.name, data, is_label_show=True, label_pos='inside', treemap_left_depth=depth)
        return treemap

    def pruning(self):
        # TODO: Pruning
        pass


def gini(x):
    """

    :param x:
    :return: gini index
    """
    x_values = list(set(x))
    p = 1
    for x_value in x_values:
        p += (x[x == x_value].shape[0]/x.shpae[0])**2
    return 1-p


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
    cols = df.columns.tolist()
    # for col in cols[:-1]:
    #     print(col, "gain", gain(df[col], df[cols[-1]]))
    #     print(col, "gain_ratio", gain_ratio(df[col], df[cols[-1]]))
    X = df[cols[:-1]].values
    y = df[cols[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Tree(eps=0.02, feas=cols, criterion="gr")
    clf.fit(X_train, y_train)
    # print(clf.tree_)
    clf.describe_tree(clf.tree_)

    print(clf.predict(X_test))
    # for x_test in X_test:
    #     print(clf.predict(x_test))

