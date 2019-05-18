# CH14 聚类方法

[TOC]

## 前言

### 章节目录

1. 聚类的基本概念
   1. 相似度或距离
   1. 类或簇
   1. 类与类之间的距离
1. 层次聚类
1. k均值聚类
   1. 模型
   1. 策略
   1. 算法
   1. 算法特性

### 导读

- Kmeans是1967年由MacQueen提出的，注意KNN也是1967年提出的，作者是Cover和Hart。
- 

## 聚类的基本概念

以下实际上是算法实现过程中的一些属性。

矩阵$X$表示样本集合，$X\in \R^m,x_i,x_j\in X, x_i=(x_{1i},x_{2i},\dots,x_{mi})^{\mathrm T},x_j=(x_{1j},x_{2j},\dots,x_{mj})^\mathrm T$，$n$个样本，每个样本是包含$m$个属性的特征向量，

### 距离或者相似度

#### 闵可夫斯基距离

$$
d_{ij}=\left(\sum_{k=1}^m|x_{ki}-x_{kj}|^p\right)^\frac{1}{p}\\
p \ge 1
$$

![fig3_2](assets/fig3_2.png)

这个图可以再展开

#### 马哈拉诺比斯距离

马氏距离

$d_{ij}=\left[(x_i-x_j)^\mathrm TS^{-1}(x_i-x_j)\right]^{\frac{1}{2}}$



#### 相关系数

$$
r_{ij}=\frac{\sum\limits_{k=1}^m(x_{ki}-\bar x_i)(x_{kj}-\bar x_j)}{\left[\sum\limits_{k=1}^m(x_{ki}-\bar x_i)^2\sum\limits_{k=1}^m(x_{kj}-\bar x_j)^2\right]^\frac{1}{2}}\\
\bar x_i=\frac{1}{m}\sum\limits_{k=1}^mx_{ki}\\
\bar x_j=\frac{1}{m}\sum\limits_{k=1}^mx_{kj}
$$



#### 夹角余弦

$$
s_{ij}=\frac{\sum\limits_{k=1}^m x_{ki}x_{kj}}{\left[\sum\limits_{k=1}^mx_{ki}^2\sum\limits_{k=1}^mx_{kj}^2\right]^\frac{1}{2}}
$$

#### 距离和相关系数的关系

其实树上的这个图，并看不出来距离和相关系数的关系。但是书中标注了角度的符号。



### 类或簇

### 类与类之间的距离

这些实际上是算法实现过程中的一些属性。

类的特征包括：均值，直径，样本散布矩阵，样本协方差矩阵

类与类之间的距离：最短距离，最长距离，中心距离，平均距离。

## 层次聚类

## Kmeans聚类





## 参考

