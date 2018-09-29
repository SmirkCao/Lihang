# CH02 感知机

[TOC]

## 前言

### 章节目录

1. 感知机模型
1. 感知机学习策略
   1. 数据集的线性可分性
   1. 感知机学习策略
   1. 感知机学习算法
1. 感知机学习算法
   1. 感知机学习算法的原始形式
   1. 算法的收敛性
   1. 感知机学习算法的对偶形式

### 导读

感知机是二类分类的线性分类模型. 

本章中涉及到向量内积，有超平面的概念，也有线性可分数据集的说明，在策略部分有说明损关于失函数的选择的考虑，可以和[CH07](../CH07/README.md)一起看。

感知机的激活函数是符号函数.

## 三要素

### 模型

输入空间：$\mathcal X\sube \bf R^n$

输出空间：$\mathcal Y={+1,-1}$

> 决策函数：$f(x)=sign (w\cdot x+b)$

### 策略

确定学习策略就是定义**(经验)**损失函数并将损失函数最小化。

注意这里提到了**经验**，所以学习是base在**训练数据集**上的操作

#### 损失函数选择

> 损失函数的一个自然选择是误分类点的总数，但是，这样的损失函数**不是参数$w,b$的连续可导函数，不易优化**
>
> 损失函数的另一个选择是误分类点到超平面$S$的总距离，这是感知机所采用的

感知机学习的经验风险函数(损失函数)
$$
L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)
$$
其中$M$是误分类点的集合

给定训练数据集$T$，损失函数$L(w,b)$是$w$和$b$的连续可导函数



### 算法

#### 原始形式

> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal X=\bf R^n\mit , y_i\in \cal Y\it =\{-1,+1\}, i=1,2,\dots,N; \ \ 0<\eta\leqslant 1$
>
> 输出：$w,b;f(x)=sign(w\cdot x+b)$
>
> 1. 选取初值$w_0,b_0$
>
> 1. 训练集中选取数据$(x_i,y_i)$
>
> 1. 如果$y_i(w\cdot x_i+b)\leqslant 0$
>    $$
>    w\leftarrow w+\eta y_ix_i \nonumber\\
>    b\leftarrow b+\eta y_i
>    $$
>
> 1. 转至(2)，直至训练集中没有误分类点

#### 对偶形式
> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal X=\bf R^n\mit , y_i\in \cal Y\it =\{-1,+1\}, i=1,2,\dots,N; \ \ 0<\eta\leqslant 1$
>
> 输出：
> $$
> \alpha ,b; f(x)=sign\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right)\nonumber\\
> \alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T
> $$
>
> 1. $\alpha \leftarrow 0,b\leftarrow 0$
>
> 1. 训练集中选取数据$(x_i,y_i)$
>
> 1. 如果$y_i\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right) \leqslant 0​$
>    $$
>    \alpha_i\leftarrow \alpha_i+\eta \nonumber\\
>    b\leftarrow b+\eta y_i
>    $$
>
> 1. 转至(2)，直至训练集中没有误分类点

**Gram matrix**

对偶形式中，训练实例仅以内积的形式出现。

为了方便可预先将训练集中的实例间的内积计算出来并以矩阵的形式存储，这个矩阵就是所谓的Gram矩阵
$$
G=[x_i\cdot x_j]_{N\times N} \nonumber
$$

## 参考

