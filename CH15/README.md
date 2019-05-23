# CH15 奇异值分解

[TOC]

## 前言

### 章节目录

1. 奇异值分解的定义与性质
   1. 定义与定理
   1. 紧奇异值分解与截断奇异值分解
   1. 几何解释
   1. 主要性质
1. 奇异值分解的计算
1. 奇异值分解与矩阵近似
   1. 弗罗贝尼乌斯范数
   1. 矩阵的最优近似
   1. 矩阵的外积展开式

### 导读

- SVD是线性代数的概念，但在统计学中有广泛应用，PCA和LSA中都有应用，本书定义为基础学习方法。
- 奇异值分解是在平方损失意义下对矩阵的最优近似，即**数据压缩**。图像存储是矩阵，那么图像也可以用SVD实现压缩。
- 奇异值分解可以扩展到Tensor
- 推荐阅读部分推荐了MIT的18.06SC，其实可以推荐下[3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)，如果有具体的哪个点不清楚，不形象，可以考虑查下
- 提到旋转或**反射变换**。关于反射变换，定点或者定直线对称，定点的叫做中心反射，定直线的叫做轴反射。

## 线性代数回顾

这部分内容主要参考3Blue1Brown中Essence of Linear Algebra，目录列举如下：

- Chapter 1: Vectors, what even are they?
- Chapter 2: Linear combinations, span and bases
- Chapter 3: Matrices as linear transformations
- Chapter 4: Matrix multiplication as composition
- Chapter 5: The determinant
- Chapter 6: Inverse matrices, column space and null space
- Chapter 7: Dot products and cross products
- Chapter 8: Cross products vis transformations
- Chapter 9: Change of basis
- Chapter 10: Eigenvectors and eigenvalues
- Chapter 11: Abstract vector spaces

### 向量

在计算机里面，向量就是一个有序的列表。

$x$轴和$y$轴的交点是原点，是整个空间的中心和所有向量的根源。向量中的数字代表从原点出发依次在每个轴上走多远，最后可以到达向量的终点。

为了把**向量和点分开**，向量通常竖着写，用方括号包围$\left[\begin{array}\\1\\2\end{array}\right]$，而点用$(1, 2)$表示。

线性代数的每个主题都围绕着向量加法和向量数乘。

向量的加法定义是唯一一个允许向量离开原点的情形。

在现在理论中，向量的形式并不重要， 箭头，一组数，函数等都可以是向量。只要向量相加和数乘的概念遵守以下规则即可，这些规则叫做公理：
$$
\begin{align}
&\overrightarrow{u} + (\overrightarrow{v} + \overrightarrow{w}) = (\overrightarrow{u} + \overrightarrow{v} )+ \overrightarrow{w}\\
&\overrightarrow{v} + \overrightarrow{w} = \overrightarrow{w} + \overrightarrow{v}\\
&There\ is\ a\ vector\ 0\ such\ that\ 0+\overrightarrow{v}= \overrightarrow{v}for\ all\ \overrightarrow{v}\\
&For\ every\ vector\ \overrightarrow{v} there\ is\ a\ vector\ -\overrightarrow{v} so\ that\ \overrightarrow{v} +(-\overrightarrow{v}) =0\\
&a(b\overrightarrow{v})=(ab)\overrightarrow{v} \\
&1\overrightarrow{v} =\overrightarrow{v} \\
&a(\overrightarrow{v} +\overrightarrow{w}) = a\overrightarrow{v} +a\overrightarrow{w} \\
&(a+b)\overrightarrow{v} =a\overrightarrow{v} +b\overrightarrow{v} 
\end{align}
$$

#### 向量加法

$$
\left[\begin{array}\\x_1\\y_1\end{array}\right]+\left[\begin{array}\\x_2\\y_2\end{array}\right]=\left[\begin{array}\\x_1+x_2\\y_1+y_2\end{array}\right]
$$

#### 向量数乘

Scaling，缩放的过程。用于缩放的数字，叫做标量，Scalar。

在线性代数中，数字的作用就是缩放向量。
$$
2\cdot\left[\begin{array}\\x\\y\end{array}\right]=\left[\begin{array}\\2x\\2y\end{array}\right]
$$

### 基

向量可以考虑成是把**基**缩放并且相加，当我们用一组数字描述向量时，他们都依赖于我们正在使用的基。

向量的线性组合与空间张成。

我们通常用向量的终点代表向量，起点位于原点。

### 线性变换

1. 直线仍然变成直线
1. 原点保持不变

保持网格平行并等距的变换，向量作为输入输出。

一个二维线性变换，仅由四个数字完全确定。$\left[\begin{array}\\i_1& j_1\\i_2&j_2\end{array}\right]$描述了线性变换。
$$
\left[\begin{array}\\i_1& j_1\\i_2&j_2\end{array}\right]
\left[\begin{array}\\x\\y\end{array}\right]
=x\underbrace{\left[\begin{array}\\i_1\\j_1\end{array}\right]}_{basis}
+y\underbrace{\left[\begin{array}\\i_2\\j_2\end{array}\right]}_{basis}
=\left[\begin{array}\\xi_1+yi_1\\xi_2+yj_2\end{array}\right]
$$
逆时针旋转90度(90 rotation counterclockwise)的线性变换矩阵，可以从x-y的基旋转之后的值来得到。
$$
\left[\begin{array}\\0&-1\\1&0\end{array}\right]
$$


Shear变换
$$
\left[\begin{array}
\\1& 0
\\1& 1
\end{array}\right]
$$
线性变换是操纵空间的一种手段，他保持网格平行等距分布，且原点保持不动。

### 矩阵乘法

矩阵乘法的几何意义是一个线性变换之后再跟一个线性变换，两个线性变换的相继作用。

顺序从右到左，因为我们函数在变量左侧。

这部分提到`Good Explanation > Symbolic proof`

二维平面的结果，可以完美的推广到三维的空间。三维线性变换由基向量的去向完全决定。

### 行列式

`The purpose of computation is insight, not numbers.-Richard Hamming`

1. 一个矩阵的行列式的绝对值为$k$说明将原来一个区域的面积变为$k$倍，变成0了说明降维了，平面压缩成了线，或者点。

   行列式为0说明降维了。

1. 行列式可以为负数，说明翻转了。这是二维空间的定向，三维空间的定向是“右手定则”

$$
det(\left[\begin{array}
\\a&c
\\b&d
\end{array}\right])=ad-bc
$$

### 线性方程组

Linear system of equations

$A\overrightarrow{x}=\overrightarrow{v}$

寻找一个向量$\overrightarrow{x}$，经过线性变换后和$\overrightarrow{v}$重合

$A^{-1}$的核心性质就是
$$
A^{-1}A=
\left[
\begin{array}
\\1&0
\\0&1
\end{array}
\right]
$$
这个也叫恒等变换。
$$
A^{-1}A\overrightarrow{x}=\overrightarrow{x}=A^{-1}\overrightarrow{v}
$$
$det(A)\neq 0\Rightarrow A^{-1} exists$

### 秩

Rank代表变换后空间的维数。

#### 列空间

矩阵的列告诉我们基向量变换之后的位置，列空间就是矩阵的列所张成的空间。

这部分实际上是书中[附录D](../APP/README.md)介绍的内容。

秩的定义是列空间的维数。

满秩，就是秩等于列数

零向量一定在列空间内，满秩变换中，唯一能落在原点的就是零向量自身。

#### 零空间

变换后，落在零向量的点的集合是零空间，或者叫核，所有可能解的集合。



### 非方阵

$3\times 2$，矩阵是把二维空间映射到三维空间上，因为矩阵有两列，说明输入空间有两个基向量，三行表示每一个基向量在变换后用三个独立的坐标来描述。

$2\times 3$，矩阵是把三维空间映射到二维空间上，因为矩阵有三列，说明输入空间有三个基向量，二行表示每一个基向量在变换后用二个独立的坐标来描述。

$1\times 2$，矩阵是把二维空间映射到一维数轴上，因为矩阵有两列，说明输入空间有两个基向量，一行表示每一个基向量在变换后用一个独立的坐标来描述。

### 叉乘

是线性的，一旦知道是线性的，就可以引入对偶性的思考了。

点乘与叉乘非常重要。

### 转移矩阵
$$
A^{-1}MA
$$
暗示着数学上的转移作用，中间的矩阵代表所见到的变换，外侧两个矩阵代表着转移作用，也就是视角上的转换。矩阵乘积仍然代表同一个变换$M$，只不过是其他人的视角。

### 特征向量与特征值

特征向量，在变换过程中留在了自己张成的空间内，这样的向量叫做特征向量。在变换过程中只受到拉伸或者压缩。

特征向量在变换中拉伸或者压缩的比例因子叫做特征值。

理解线性变换的作用的关键往往较少依赖于你的特定坐标系。
$$
\begin{aligned}
A\overrightarrow{v}&=(\lambda I )\overrightarrow{v}\\
A\overrightarrow{v} - (\lambda I )\overrightarrow{v}&=0\\
(A - \lambda I )\overrightarrow{v}&=0\\
det(A-\lambda I) &= 0
\end{aligned}
$$
这部分，行列式为0的几何意义很重要。如果没有实数解，说明没有特征向量。

### 空间

Determinant and eigenvectors don't care about the coordinate system.
行列式告诉你一个变换对面积的缩放比例，特征向量则是在变换中保留在他所张成的空间中的向量，这两者都是暗含与空间中的性质，坐标系的选择并不会改变他们最根本的值。

函数实际上只是另一种向量。
- 函数的线性变换，比如微积分中的导数，有时候会用**算子**来表示**变换**的意思。
- 求导具有可加性和成比例性。
- 函数空间趋近于无限维
- 多项式空间，求导
- 矩阵向量乘法和矩阵求导看起来是不相关的，但实际上是一家人。

| 线性代数 | 函数 |
| ---- | ---- |
| 线性变换|线性操作|
|点乘|内积|
|特征向量|特征函数|

只要处理的对象有合理的数乘和相加的概念，只要定义满足公理，就能应用线性代数中的结论。





抽象性带来的好处是我们能得到一般性的结论。

最后补充一点，关于视频中描述一个变换的中间步骤的过程，可以参考[Chapter 14](https://www.youtube.com/watch?v=PFDu9oVAE-g&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=14)的7:45左右的视频内容体会。



## 奇异值分解定义与性质

矩阵的奇异值分解是指将$m\times n$实矩阵$A$表示为以下三个实矩阵乘积形式的运算
$$
A=U\mit\Sigma V^\mathrm T
$$



## 奇异值分解与矩阵近似

奇异值分解也是一种矩阵近似的方法，这个近似是在弗罗贝尼斯范数意义下的近似。

矩阵的弗罗贝尼斯范数是向量的$L_2$范数的直接推广，对应着机器学习里面的平方损失函数。矩阵范数(matrix norm)也是一个很大的概念，详细内容可以扩展下[^1]。

## 习题

### 15.5

这个例子很有意思，实际上这个图，是一种稀疏表示，而矩阵是一种稠密的表示，但是这个矩阵也是一个稀疏矩阵。

## 参考

[^1]:  [矩阵范数](https://zh.wikipedia.org/wiki/矩陣範數)

