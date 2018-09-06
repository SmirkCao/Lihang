# CH9 EM算法及其推广

## 前言

书中这个章节的结构

1. EM算法的引入
   1. EM算法
   1. EM算法的导出
   1. EM算法在非监督学习中的应用
1. EM算法的收敛性
1. EM算法在高斯混合模型学习中的应用
   1. 高斯混合模型
   1. 高斯混合模型参数估计的EM算法
1. EM算法的推广
   1. F函数的极大极大算法

EM算法可以用于生成模型的非监督学习, EM算法是个一般方法, 不具有具体模型.

## 符号说明

> 一般地, 用$Y$表示观测随机变量的数据, $Z$表示隐随机变量的数据. $Y$和$Z$一起称为**完全数据**(complete-data), 观测数据$Y$又称为**不完全数据**(incomplete-data)

上面这个概念很重要, Dempster在1977年提出EM算法的时候文章题目就是<Maximum likelihood from incomplete data vi the EM algorithm>, 具体看书中本章参考文献[1]

>假设给定观测数据$Y$, 其概率分布是$P(Y|\theta)$, 其中$\theta$是需要估计的模型参数
>那么不完全数据$Y$的似然函数是$P(Y|\theta)$, 对数似然函数是$L(\theta)=\log P(Y|\theta)$
>
>假设$Y$和$Z$的联合概率分布是$P(Y,Z|\theta)$, 那么完全数据的对数似然函数是$\log P(Y,Z|\theta)$

上面这部分简单对应一下

|                   |                                       |                       |                                  |
| ----------------- | ------------------------------------- | --------------------- | -------------------------------- |
|                   | 观测数据$Y$                           | 不完全数据$Y$         |                                  |
| 不完全数据$Y$     | 概率分布$P(Y|\theta)$                 | 似然函数$P(Y|\theta)$ | 对数似然函数$ \log P(Y|\theta)$  |
| 完全数据 $(Y, Z)$ | $Y$和$Z$的联合概率分布$P(Y,Z|\theta)$ |                       | 对数似然函数$\log P(Y,Z|\theta)$ |

观测数据$Y$

有一点要注意下, 这里没有出现$X$, 在**9.1.3**节中有提到一种理解

> - 有时训练数据只有输入没有对应的输出${(x_1,\cdot),(x_2,\cdot),\dots,(x_N,\cdot)}$, 从这样的数据学习模型称为非监督学习问题.
> - EM算法可以用于生成模型的非监督学习.
> - 生成模型由联合概率分布$P(X,Y)$表示, 可以认为非监督学习训练数据是联合概率分布产生的数据. $X$为观测数据, $Y$为未观测数据.

有时候, 只观测显变量看不到关系, 就需要把隐变量引进来.

## 问题描述

书中用例子来介绍EM算法的问题, 并给出了EM算法迭代求解的过程, 具体例子描述见**例9.1**,解的过程记录在这里. 

三硬币模型可以写作
$$
\begin{equation}
\begin{aligned}
P(y|\theta)&=\sum_z P(y,z|\theta) \\
&=\sum_z P(z|\theta)P(y|z,\theta) \\
&=\pi p^y (1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}
\end{aligned}
\end{equation}
$$
以上

1. 随机变量$y$是观测变量, 表示一次试验观测的结果是1或0
1. 随机变量$z$是隐变量, 表示未观测到的掷硬币$A$的结果
1. $\theta=(\pi,p,q)$是模型参数
1. 这个模型是**以上数据**(1,1,0,1,0,0,1,0,1,1)的生成模型.



观测数据表示为$Y=(Y_1, Y_2, Y_3, \dots, Y_n)^T$, 未观测数据表示为$Z=(Z_1,Z_2, Z_3,\dots, Z_n)^T$, 则观测数据的似然函数为
$$
P(Y|\theta) = \sum\limits_{Z}P(Z|\theta)P(Y|Z,\theta)
$$
即
$$
P(Y|\theta)=\prod\limits^{n}_{j=1}[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$
考虑求模型参数$\theta=(\pi,p,q)$的极大似然估计, 即
$$
\hat \theta = \arg\max\limits_{\theta}\log P(Y|\theta)
$$


## EM算法的解释

注意这里EM不是模型, 是个一般方法, 不具有具体的模型.

### PRML

$kmeans \rightarrow GMM \rightarrow EM$

### 统计学习方法

1. $MLE \rightarrow B$
1. $F$函数的极大-极大算法

## 高斯混合模型

**混合模型**, 有多种, 高斯混合模型是最常用的.

高斯混合模型(Gaussian Mixture Model)是具有如下概率分布的模型:
$$
P(y|\theta)=\sum\limits^{K}_{k=1}\alpha_k\phi(y|\theta_k)
$$
其中, $\alpha_k$是系数, $\alpha_k\ge0$, $\sum\limits^{K}_{k=1}\alpha_k=1$, $\phi(y|\theta_k)$是高斯分布密度, $\theta_k=(\mu,\sigma^2)$
$$
\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-\mu_k)^2}{2\sigma_k^2}\right)
$$
上式表示第k个分模型.

高斯混合模型的参数估计是EM算法的一个重要应用, 隐马尔科夫模型的非监督学习也是EM算法的一个重要应用.

## 广义期望极大

广义期望极大(generalized expectation maximization, $GEM$)

## 其他

1. 关于习题9.3 
   GMM模型的参数($\alpha _k, \mu _k, \sigma^2_k $)应该是$3k$个, 题目9.3中提出两个分量的高斯混合模型的5个参数, 是因为参数$\alpha_k$满足$\sum_{k=1}^K\alpha _k=1$ 
1. 

## 参考

1. 
2. [Sklearn Gaussian Mixed Model](http://scikit-learn.org/stable/modules/mixture.html)

