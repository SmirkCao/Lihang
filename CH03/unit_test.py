# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: unit_test
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>
import unittest
import knn


class TestStringMethods(unittest.TestCase):

    def test_e31(self):
        x = [[1, 1], [5, 1], [4, 4]]
        y = [1, 2, 3]
        rst = []
        for p in range(1, 5):
            clf_knn = knn.KNN(k=1, p=p)
            clf_knn.fit(x[1:])
            rst.extend(clf_knn.predict([x[0]]))

        self.assertEqual(rst, [[2], [2], [2], [2]])

    def test_e32(self):
        x = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
        y = [1, 2, 3, 4, 5, 6]
        clf_knn = knn.KNN(k=1, p=2)
        clf_knn.fit(x_=x)
        self.assertEqual(clf_knn.kdtree, ([7, 2],
                                          ([5, 4],
                                           ([2, 3], None, None),
                                           ([4, 7], None, None)),
                                          ([9, 6],
                                           ([8, 1], None, None),
                                           None)))

    def test_e33(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [1, 2, 3, 4, 5, 6, 7]

        pass


if __name__ == '__main__':
    unittest.main()
