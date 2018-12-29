# -*-coding:utf-8-*-
# Project: Lihang
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>
from adaboost import *
import pandas as pd
import numpy as np
import unittest


class TestAdaBoost(unittest.TestCase):
    @staticmethod
    def load_data(path_="./Input/data_8-1.txt"):
        # p140, ex8.1 -> data_8-1.txt
        # p149, ex8.2 -> data_8-2.txt
        df = pd.read_csv(path_)
        x = df["x"].values
        y = df["y"].values
        return x, y

    # @unittest.skip("")
    def test_adaboost_algo(self):
        # ex8.1
        x, y = self.load_data()
        fs = [clf_great_than_, clf_less_than_]

        y1 = y.copy()
        print("----clf1----")
        clf1 = BiSection()
        clf1.fs = fs
        d1 = np.ones(x.size) / x.size
        d1 = np.round(d1, 5)
        print("d1=", d1)
        v1, fv1, err_his_f1 = clf1.fit(x, y1, d_=d1)
        print("v1=", v1, "fv1", fv1)
        G1 = clf1.predict(x)
        print("G1=", G1)
        e1 = np.sum(d1[G1 != y])
        e1 = np.round(e1, 4)
        print("e1=", e1)
        alpha1 = np.log((1 - e1) / e1) / 2
        alpha1 = np.round(alpha1, 4)
        print("alpha1=", alpha1)
        d2 = d1 * np.exp(-alpha1 * y * G1) / np.sum(d1 * np.exp(-alpha1 * y * G1))
        d2 = np.round(d2, 5)
        print("d2=", d2)
        f1 = alpha1 * clf1.predict(x)
        f1 = np.round(f1, 4)
        print("f1=", f1)
        sign_f1 = np.sign(alpha1 * clf1.predict(x))
        print("sign_f1=", sign_f1)
        acc1 = accuracy_score(sign_f1, y)
        print("acc1=", acc1)
        print("----clf2----")
        clf2 = BiSection()
        clf2.fs = fs
        v2, fv2, err_his_f2 = clf2.fit(x, y, d_=d2)
        print("v2=", v2)
        G2 = clf2.predict(x)
        print("G2=", G2)
        e2 = np.sum(d2[G2 != y])
        e2 = np.round(e2, 4)
        print("e2=", e2)
        alpha2 = np.log((1 - e2) / e2) / 2
        alpha2 = np.round(alpha2, 4)
        print("alpha2=", alpha2)
        d3 = d2 * np.exp(-alpha2 * y * G2) / np.sum(d2 * np.exp(-alpha2 * y * G2))
        d3 = np.round(d3, 4)
        print("d3=", d3)
        f2 = alpha1 * clf1.predict(x) + alpha2 * clf2.predict(x)
        f2 = np.round(f2, 4)
        print("f2=", f2)
        sign_f2 = np.sign(alpha1 * clf1.predict(x) + alpha2 * clf2.predict(x))
        print("sign_f2=", sign_f2)
        acc2 = accuracy_score(sign_f2, y)
        print("acc2=", acc2)
        print("----clf3----")
        clf3 = BiSection()
        clf3.fs = fs
        v3, fv3, err_his_f3 = clf3.fit(x, y, d_=d3)
        print("v3=", v3)
        G3 = clf3.predict(x)
        print("G3=", G3)
        e3 = np.sum(d3[G3 != y])
        e3 = np.round(e3, 4)
        print("e3=", e3)
        alpha3 = np.log((1 - e3) / e3) / 2
        alpha3 = np.round(alpha3, 4)
        print("alpha3=", alpha3)
        d4 = d3 * np.exp(-alpha3 * y * G3) / np.sum(d3 * np.exp(-alpha3 * y * G3))
        d4 = np.round(d4, 4)
        print("d4=", d4)
        f3 = alpha3 * clf3.predict(x) + alpha2 * clf2.predict(x) + alpha1 * clf1.predict(x)
        f3 = np.round(f3, 4)
        print("f3=", f3)
        sign_f3 = np.sign(f3)
        print("sign_f3=", sign_f3)
        acc3 = accuracy_score(sign_f3, y)
        print("acc3=", acc3)

        # for markdown table
        def to_table(x_, name_):
            return "|"+name_+"|"+"|".join(map(str, x_.tolist()))+"|"

        print(to_table(d1, "d1"))
        print(to_table(G1, "G1"))
        print(to_table(d2, "d2"))
        print(to_table(f1, "f1"))
        print(to_table(sign_f1, "sign_f1"))
        print(to_table(G2, "G2"))
        print(to_table(d3, "d3"))
        print(to_table(f2, "f2"))
        print(to_table(sign_f2, "sign_f2"))
        print(to_table(G3, "G3"))
        print(to_table(d4, "d4"))
        print(to_table(f3, "f3"))
        print(to_table(sign_f3, "sign_f3"))

    # @unittest.skip(":")
    def test_adaboost(self):
        x, y = TestAdaBoost.load_data()

        clf = AdaBoost(BiSection, max_iter=3)
        clf.fs = [clf_great_than_, clf_less_than_]
        clf.fit(x, y)
        y_pred = clf.predict(x)
        print("final accuracy : ", accuracy_score(y_pred, y))

    def test_e82(self):
        # ex8.2
        x, y = TestAdaBoost.load_data(path_="./Input/data_8-2.txt")
        rgs = AdaBoostRegressor(max_iter=6)
        rgs.fit(x, y)
        print(rgs)


if __name__ == '__main__':
    unittest.main()
