# CH18 概率潜在语义分析

[TOC]

## 前言

### 章节目录

1. 概率潜在语义分析模型
   1. 基本想法
   1. 生成模型
   1. 共现模型
   1. 模型性质
1. 概率潜在语义分析的算法

### 导读

- 这章就是按照三要素的顺序来讲的：模型，策略，算法
- Use a probabilistic method instead of SVD to tackle the problem[^1]，一个具体的概率分布的存储形式就是矩阵，能用矩阵处理的，都能弄到数据库里。
- 概率潜在语义分析受潜在语义分析启发，两者可以通过**矩阵分解**关联起来，矩阵分解是个框架，在这个框架下有很多算法。
- 模型的最大特点是用隐变量表示话题。其实之前的潜在语义分析应该也是用隐变量表示话题吧？所有的话题模型，都基于同样的假设[^1]：1. 文本由话题组成；2 话题由单词组成。
- 一个话题表示一个语义内容，这部分内容算是NLU(nature language understanding)，在检索情况下，提供一个检索词，可以使用近义词集合去寻找。
- LSA可以解决多词一义(同义词)的问题，PLSA可以解决一词多义的问题。
- 生成模型和共现模型在概率公式意义上是等价的，生成模型刻画文本-单词共现数据生成的过程，共现模型描述文本-单词共现数据拥有的模式。
- 讨论模型参数的部分，和DL里面参数实际上是一样的。好好看下图18.4，对应了生成模型的参数情况。参数数量的说明也表示了你需要用一个什么样的数组来保存这些关系。参数，算模型的性质。
- EM算法的核心在是定义$Q$函数
- 在[第九章](../CH09/README.md)中就有专门小节提到EM算法在无监督学习中的应用，EM算法可以用在生成模型的无监督学习，这一章诠释了这一点。
- 单词-文本矩阵就是文本-单词共现数据。
- 参考文献1是1999年发表的，文中有提到data-driven，1999年就数据驱动了，20年过去了。参考文献2里面提到了LSA存在一些不足，`mainly due to its unsatisfactory statistical foundation.`然后提到PLSA`has solid statistical foundation.`
- 注意在[^1]中有提到PLSA中概率与SVD中奇异向量的对应关系，和书中的描述是相反的。但是在书中的参考文献2中，Hofmann有说明这个对应关系，和书中描述一致。

## 内容

### 模型
#### 生成模型
1. 依据概率分布$P(d)$从文本集合中选取一个文本$d$，共生成$N$个文本，针对每个文本执行以下操作：
1. 文本$d$给定的条件下，依据条件概率分布$P(z|d)$，从话题集合随机选取一个单词$w$，生成$L$个话题，这里$L$是文本长度。
1. 在话题$z$给定的条件下，依据条件概率分布$P(w|z)$，从单词集合中随机选取一个单词$w$

模型生成的是单词-话题-文本三元组$(w,z,d)$，观测到的是$(w,d)$二元组的集合。观测数据表示为单词文本矩阵$T$的形式，行表示单词，列表示文本，元素表示单词-文本对$(w,d)$的出现次数。
$P(T)=\prod\limits_{(w,d)}P(w,d)^{n(w,d)}$，其中$n(w,d)$表示$(w,d)$出现的次数，这个$n$的作用可以和前面[LR部分](../CH06/README.md)中$f^\#$对比看，差不多。
$$
\begin{aligned}
P(w,d)&=P(d)P(w|d)\\
&=P(d)\sum_z \color{red}P(w,z|d)\\
&=P(d)\sum_z \color{red}P(w|z)P(z|d)
\end{aligned}
$$
以上红色部分来自iid假设，在给定$z$条件下单词$w$和文本$d$条件独立。

#### 共现模型

首先有话题的概率分布，然后有话题给定条件下文本的条件概率分布，以及话题给定条件下单词的条件概率分布。

共现模型，联合概率分布。
$$
\begin{aligned}
P(w,d)
&=\sum\limits_{z\in Z}P(z)\color{red}P(w,d|z)\\
&=\sum\limits_{z\in Z}P(z)\color{red}P(w|z)P(d|z)
\end{aligned}
$$
以上红色部分来自iid假设，在给定$z$的条件下，单词$w$和文本$d$是条件独立的，这个相对好理解一点。
#### 与潜在语义分析的关系

$$
\begin{aligned}
X^\prime&=U^\prime\mit{\Sigma}^\prime V^{\prime\mathrm{T}}\\
X^\prime&=[P(w,d)]_{M\times N}\\
U^\prime&=[P(w|z)]_{M\times K}\\
\mit\Sigma^\prime&=[P(z)]_{K\times K}\\
V^\prime&=[P(d|z)]_{N\times K}
\end{aligned}
$$

概率潜在语义分析模型中$U^\prime$和$V^\prime$是非负的、规范化的，表示条件概率分布。
潜在语义分析模型中$U$和$V$是正交的，未必非负，并不表示概率分布。

### 策略
极大似然估计

### 算法

生成模型的EM算法
**已知：**
单词集合$W=\{w_1, w_2, \cdots, w_M\}$
文本集合$D=\{d_1, d_2, \cdots, d_N\}$
话题集合$Z=\{z_1, z_2, \cdots, z_K\}$
共现数据$T={n(w_i, d_j)}, i=1,2,\cdots, M, j=1,2,\cdots,N$
**求解：**
概率潜在语义分析模型(生成模型)的参数。

对数似然函数
$$
\begin{aligned}
L&=\sum_{i=1}^M\sum_{j=1}^N n(w_i,d_j) \log P(w_i,d_j)\\
&=\sum_{i=1}^M\sum_{j=1}^N  n(w_i,d_j) \log \left[\sum_{k=1}^K P(w_i|z_k)P(z_k|d_j)\right]
\end{aligned}
$$

E步：计算$Q$函数
M步：极大化$Q$函数

**算法18.1** 概率潜在语义模型参数估计的EM算法

1. 设置参数$P(w_i|z_k)$和$P(z_k|d_j)$的初始值
1. 迭代执行E和M步骤
1. E步：
$$
P(z_k|w_i,d_j)=\frac{P(w_i|z_k)P(z_k|d_j)}{\sum_{k=1}^K P(w_i|z_k)P(z_k|d_j)}
$$
4. M步：
$$
\begin{aligned}
P(w_i|z_k)&=
\frac
{\sum_{j=1}^N n(w_i,d_j)P(z_k|w_i,d_j)}
{\sum_{m=1}^M\sum_{j=1}^N n(w_m,d_j)P(z_k|w_m,d_j)}\\
P(z_k|d_j)&=\frac{\sum_{i=1}^M n(w_i,d_j) P(z_k|w_i, d_j)}{n(d_j)}
\end{aligned}
$$



## 参考

[^1]: [Topic Modeling with LSA, PLSA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)