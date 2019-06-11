#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 5/23/19
# Author: 😏 <smirk dot cao at gmail dot com>

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys


class TestSVDMethods(unittest.TestCase):

    def test_e_15_1(self):
        A = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 4],
                      [0, 3, 0, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, 0]])
                      
        u, s, vh = np.linalg.svd(A)
        ur, sr, vrh = np.linalg.svd(A, full_matrices=False)
        r = np.linalg.matrix_rank(A)
        print("\n")
        print(u)
        # 注意s和A.T*T
        # 奇异值，特征值的平方根
        print(s)
        # n阶对称实矩阵
        print(np.dot(A.T, A))
        print(vh)
        print(40*"*")
        print(ur)
        print(sr)
        print(vrh)
        print(40*"*")
        print(r)
    
    def test_e_15_4_e(self):
        import matplotlib.pyplot as plt
        import sys

        size = 3
        alpha = 0.6
        head_width = 0.02
        head_length = 0.05
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(1, 1, 1, aspect=1)

        circle1 = plt.Circle((0.5, 0.5), 0.3,
                             color="pink", fill=True, alpha=alpha)
        ax.add_artist(circle1)
        plt.arrow(0.5, 0.5, 0.0, 0.3-head_length, fc="r", ec="r",
                  alpha=alpha, head_width=head_width, head_length=head_length)
        plt.arrow(0.5, 0.5, 0.3-head_length, 0, fc="y", ec="y",
                  alpha=alpha, head_width=head_width, head_length=head_length)

        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        folder = sys.path[0]
        plt.savefig(folder+"/temp.png")
        img = plt.imread(folder+"/temp.png")

        print(img.shape)
        A = np.array([[3, 1],
                      [2, 1]])
        u, s, vh = np.linalg.svd(A)
        print(40*"*")
        print(u)
        print(s)
        print(vh)
        print(40*"*")

        rst = np.zeros_like(img)
        print(rst.shape)
        for i in np.arange(img.shape[0]):
            for j in np.arange(img.shape[1]):
                i_, j_ = np.diagonal(vh*np.array([[i - rst.shape[0]/2, 0],
                                                  [0, j - rst.shape[1]/2]]))

                # print(i, j, ":", int(i_+150), int(j_+150))
                idx, idy = int(i_+150), int(j_+150)
                if idx < 300 and idy < 300 and idx >= 0 and idy >= 0:
                    rst[idx, idy] = img[i, j]
        plt.clf()
        plt.axis('off')
        plt.imshow(rst)
        plt.savefig(folder+"/temp_vh.png")

        rst_s = np.zeros_like(img)

        for i in np.arange(img.shape[0]):
            for j in np.arange(img.shape[1]):
                i_, j_ = np.diagonal(s*np.array([[i - rst.shape[0]/2, 0],
                                                [0, j - rst.shape[1]/2]]))
                
                idx, idy = int(i_+150), int(j_+150)
                if idx < 300 and idy < 300 and idx >= 0 and idy >= 0:
                    rst_s[idx, idy] = rst[i, j]
        plt.clf()
        plt.axis('off')
        plt.imshow(rst_s)
        plt.savefig(folder+"/temp_svh.png")
        rst_a = np.zeros_like(img)

        for i in np.arange(img.shape[0]):
            for j in np.arange(img.shape[1]):
                i_, j_ = np.diagonal(u*np.array([[i - rst.shape[0]/2, 0],
                                                [0, j - rst.shape[1]/2]]))

                idx, idy = int(i_+150), int(j_+150)
                if idx < 300 and idy < 300 and idx >= 0 and idy >= 0:
                    rst_a[idx, idy] = rst_s[i, j]
        plt.clf()
        plt.axis('off')
        plt.imshow(rst_a)
        plt.savefig(folder+"/temp_usvh.png")

        # plt.imshow(rst_a)
        plt.show()

    @staticmethod
    def draw_arrow(origin, e1, e2):
        size = 3
        alpha = 0.6
        head_width = 0.2
        head_length = 0.3
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(1, 1, 1, aspect=1)
        circle1 = plt.Circle((0.5, 0.5), 0.5,
                             color="pink", fill=True, alpha=alpha)
        ax.add_artist(circle1)
        plt.arrow(origin[0], origin[1], e1[0], e1[1], fc="r", ec="r",
                  alpha=alpha, head_width=head_width, head_length=head_length)
        plt.arrow(origin[0], origin[1], e2[0], e2[1], fc="y", ec="y",
                  alpha=alpha, head_width=head_width, head_length=head_length)

        plt.axis('off')
        return fig

    def test_e_15_4(self):

        e1 = np.array([1, 0])
        e2 = np.array([0, 1])

        origin = np.array([0.5, 0.5])
        fig = TestSVDMethods.draw_arrow(origin, e1, e2)
        folder = sys.path[0]
        plt.savefig(folder+"/e15-4.png")
        img = plt.imread(folder+"/e15-4.png")
        plt.axis('on')
        plt.imshow(img)
        plt.show()

        A = np.array([[3, 1],
                      [2, 1]])
        u, s, vh = np.linalg.svd(A)
        arr1 = np.dot(vh, e1)
        arr2 = np.dot(vh, e2)
        fig = TestSVDMethods.draw_arrow(origin, arr1, arr2)
        folder = sys.path[0]
        plt.savefig(folder+"/e15-4_vh.png")
        img = plt.imread(folder+"/e15-4_vh.png")
        plt.axis('on')
        plt.imshow(img)
        plt.show()

        arr1 = np.dot(s*np.eye(2), arr1)
        arr2 = np.dot(s*np.eye(2), arr2)
        fig = TestSVDMethods.draw_arrow(origin, arr1, arr2)
        folder = sys.path[0]
        plt.savefig(folder+"/e15-4_svh.png")
        img = plt.imread(folder+"/e15-4_svh.png")
        plt.axis('on')
        plt.imshow(img)
        plt.show()

        arr1 = np.dot(u, arr1)
        arr2 = np.dot(u, arr2)
        fig = TestSVDMethods.draw_arrow(origin, arr1, arr2)
        folder = sys.path[0]
        plt.savefig(folder+"/e15-4_usvh.png")
        img = plt.imread(folder+"/e15-4_usvh.png")
        plt.axis('on')
        plt.imshow(img)
        plt.show()

        print(40*"*"+"u"+40*"*")
        print(u)
        print(40*"*"+"s"+40*"*")
        print(s)
        print(s*np.eye(2))
        print(40*"*"+"vh"+40*"*")
        print(vh)
        print(40*"*"+"ATA"+40*"*")
        print(np.linalg.eig(np.dot(A.T, A)))
        print(np.linalg.eigvals(np.dot(A.T, A)))
        print(np.sqrt(np.linalg.eigvals(np.dot(A.T, A))))

        print(40*"*")

        rst = np.zeros_like(img)
        print(vh)
        vhe1 = np.dot(vh, e1)
        vhe2 = np.dot(vh, e2)
        print(40*"*"+"vhe"+40*"*")

        print(vhe1)
        print(vhe2)
        svhe1 = np.dot(s*np.eye(2), vhe1)
        svhe2 = np.dot(s*np.eye(2), vhe2)
        print(40*"*"+"svhe"+40*"*")

        print(svhe1)
        print(svhe2)
        print(40*"*"+"usvhe"+40*"*")
        usvhe1 = np.dot(u, svhe1)
        usvhe2 = np.dot(u, svhe2)
        print(usvhe1)
        print(usvhe2)

    def test_e_15_5(self):
        A = np.array([[1, 1], [2, 2], [0, 0]])
        print(40*"*"+"A"+40*"*")
        print(A)
        print(40*"*"+"ATA"+40*"*")
        print(np.dot(A.T, A))
        print(40*"*"+"Eig Vector"+40*"*")
        print(np.linalg.eig(np.dot(A.T, A)))

    def test_e_15_6(self):
        A = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 4],
                      [0, 3, 0, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, 0]])
        u, s, vh = np.linalg.svd(A)
        print(u)
        print(s)
        print(vh)

        print(u[:, 0], u[:, 1], vh[0], vh[1])
        print(np.dot(u[:, 0].reshape(-1, 1), vh[0].reshape(1, -1)))
        print(np.dot(u[:, 1].reshape(-1, 1), vh[1].reshape(1, -1)))
        # 非方阵求特征值
        print(40*"*"+"ATA Eig Vector"+40*"*")
        print(np.linalg.eigh(np.dot(A.T, A)))
        print(40*"*"+"AAT Eig Vector"+40*"*")
        print(np.linalg.eigh(np.dot(A, A.T)))

    def test_t_15_5(self):
        A = np.array([[0, 20, 5, 0, 0],
                      [10, 0, 0, 3, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0]])
        u, s, vh = np.linalg.svd(A)
        print("\n")
        print(u)
        print(s)
        print(vh)
        print(vh.T)


if __name__ == "main":
    unittest.main()
