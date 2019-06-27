# 附录
![Hits](https://www.smirkcao.info/hit_gits/Lihang/APP/README.md)

这部分大概过一下算法，AB针对无约束最优化问题，C对应约束最优化问题。

## A 梯度下降法
这部分内容是介绍梯度下降, 在NN中用到最多的是SGD, 为什么不介绍SGD?

SGD，S来自与样本的随机。DL中样本很多，通常会分Batch，每个Batch刷的过程就是SGD。实际上数据量小的时候，SGD和GD一样。所以数据的Shuffle就很重要。

## B 牛顿法和拟牛顿法

### BFGS



## C 拉格朗日对偶性

在[3Blue1Brown](https://www.youtube.com/watch?v=LyGKycYT2v0&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=9)中有这样一段描述来说明对偶
$$
Duality\Leftrightarrow Natural-but-Surprising\ correspondence.
$$


在约束最优化问题中，常常利用拉格朗日对偶性将原问题转化为对偶问题，通过求解对偶问题得到原始问题的解。

为什么要这么做在[CH07](../CH07/README.md)中有说明`这样做的优点，一是对偶问题往往更容易求解；二是自然引入核函数，进而推广到非线性分类问题`

### 原始问题



### 对偶问题



## D 矩阵的基本子空间



### 向量空间的子空间

线性组合

### 向量空间的基和维数

在张成的基础上多了个线性无关的约束

### 矩阵的行空间和列空间

注意这里稍微有点，绕。
$A_{m\times n}$，$m$行，$n$列，每一行都有$n$列，所以说可以看成是$\mathrm{R}^n$的向量。

向量空间的基的个数即向量空间的维数。

### 矩阵的零空间

$N(A)=\{x\in \mathbf{R}^n|Ax=0\}$

一个矩阵的零空间的维数称为矩阵的**零度**

秩-零度定理：设$A$为一$m\times n$矩阵，则$A$的秩与$A$的零度之和为$n$。

### 子空间的正交补

$Y$是$\mathbf{R}^n$的子空间，则$Y^\perp$也是$\mathbf{R}^n$的子空间。

### 矩阵的基本子空间

矩阵代表了一种线性变换。
矩阵$A$有四个基本子空间：列空间，行空间，零空间，$A$的转置零空间(左零空间)
$$
R(A)=\{z\in \mathbf{R}^m|\exist x\in \mathbf{R}^n, z=Ax\}=C(A)
$$

$$
R(A^\mathrm{T})=\{y\in \mathbf{R}^n|\exist x\in \mathbf{R}^m, y=A^\mathrm{T}x\}=C(A^\mathrm{T})
$$

这部分在Strang的书里有四个子空间的关系图，和书中给的差不多。


## E KL散度的定义和狄利克雷分布的性质

KL散度是非对称的，也不满足三角不等式，不是严格意义的距离度量。

狄利克雷分布属于指数族分布