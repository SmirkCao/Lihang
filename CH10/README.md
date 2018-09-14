# CH10 隐马尔科夫模型

## 前言

本章目录结构如下

1. 隐马尔可夫模型的基本概念
   1. 隐马尔可夫模型的定义
   1. 观测序列的生成过程
   1. 隐马尔可夫模型的三个基本问题
1. 概率计算方法
   1. 直接计算法
   1. 前向算法
   1. 后向算法
   1. 一些概率与期望值的计算
1. 学习算法
   1. 监督学习方法
   1. Baum-Welch算法
   1. Baum-Welch模型参数估计公式
1. 预测算法
   1. 近似算法
   1. 维特比算法

动态贝叶斯网络的最简单实现隐马尔可夫模型. HMM可以看成是一种推广的混合模型.

序列化建模, 打破了数据独立同分布的假设.

有些关系需要理清

```mermaid
graph TB
	st(图模型)-->op1[有向图]
	st(图模型)-->op2[无向图]
	op1-->op3[静态贝叶斯网络]
	op1-->op4[动态贝叶斯网络]
	op2-->op5[马尔可夫网络]
	op5-->op6[吉布斯/玻尔兹曼机]
	op5-->op7[条件随机场]
	op4-->op8[隐马尔可夫模型]
	op4-->op9[卡尔曼滤波器]

```

另外一个图

```mermaid
graph TD
	subgraph 贝叶斯网络
	A1((A))-->B1((B))
	A1-->C1((C))
	B1-->D1((D))
	C1-->D1
	end
	
	subgraph 马尔可夫网络
	A2((A))---B2((B))
	A2---C2((C))
	B2---D2((D))
	C2---D2
	end
```



## 概念

有些基本的概念, 引用吴军在数学之美[^1]之中的描述. 

### 随机变量与随机过程

>19世纪, 概率论的发展从对(相对静态的)**随机变量**的研究发展到对随机变量的时间序列$s_1,s_2, s_3, \dots,s_t,\dots$,即**随机过程**(动态的)的研究
>
>数学之美,吴军

### 马尔可夫链

>随机过程有两个维度的不确定性. 马尔可夫为了简化问题, 提出了一种简化的假设, 即随机过程中各个状态$s_t$的概率分布, 只与它的前一个状态$s_{t-1}$有关, 即$P(s_t|s_1, s_2, s_3, \dots,s_{t-1})=P(s_t|s_{t-1})$
>
>这个假设后来被称为**马尔可夫假设**,而符合这个假设的随机过程则称为**马尔可夫过程**, 也称为**马尔可夫链**.
>
>数学之美, 吴军

$$
P(s_t|s_1, s_2, s_3, \dots,s_{t-1})=P(s_t|s_{t-1})
$$

时间和状态取值都是离散的马尔可夫过程也称为马尔可夫链.

### 隐含马尔可夫模型

$$
P(s_1,s_2,s_3,\dots,o_1,o_2,o_3,\dots)=\prod_tP(s_t|s_{t-1})\cdot P(o_t|s_t)
$$

隐含的是**状态**$s$

隐含马尔可夫模型由**初始概率分布**(向量$\pi$), **状态转移概率分布**(矩阵$A$)以及**观测概率分布**(矩阵$B$)确定.

隐马尔可夫模型$\lambda$ 可以用三元符号表示, 即
$$
\lambda = (A, B, \pi)
$$

其中$A,B,\pi$称为模型三要素.

 

### 两个基本假设

1. 齐次马尔科夫假设(**状态**)
1. 观测独立性假设(**观测**)

### 三个基本问题

1. 概率计算问题
   输入: 模型$\lambda=(A,B,\pi)$, 观测序列$O=(o_1,o_2,\dots,o_T)$
   输出: $P(O|\lambda)$
1. 学习问题
   输入: 观测序列 $O=(o_1,o_2,\dots,o_T)$
   输出: 输出$\lambda=(A,B,\pi)$
1. 预测问题, 也称为解码问题
   输入: 模型$\lambda=(A,B,\pi)$, 观测序列$O=(o_1,o_2,\dots,o_T)$ 
   输出: 状态序列 $I=(i_1,i_2,\dots,i_T)$

>There are three fundamental problems for HMMs:
>
>- Given the model parameters and observed data, estimate the optimal sequence of hidden states.
>- Given the model parameters and observed data, calculate the likelihood of the data.
>- Given just the observed data, estimate the model parameters.
>
>The first and the second problem can be solved by the dynamic programming algorithms known as the Viterbi algorithm and the Forward-Backward algorithm, respectively. The last one can be solved by an iterative Expectation-Maximization (EM) algorithm, known as the Baum-Welch algorithm.
>
>---hhmlearn

### 前向与后向

>求解的都是**观测序列概率**
>
>给定马尔可夫模型$\lambda$, 定义到时刻$t$部分观测序列为$o_1, o_2, \dots ,o_t$, 且状态$q_i$的概率为**前向概率**, 记作
>$$
>\alpha_t(i)=P(o_1,o_2,\dots,o_t,i_t=q_i|\lambda)
>$$
>给定马尔可夫模型$\lambda$, 定义到时刻$t$状态为$q_i$的条件下, 从$t+1$到$T$的部分观测序列为$o_{t+1}, o_{t+2}, \dots ,o_T$的概率为**后向概率**, 记作
>$$
>\beta_t(i)=P(o_{t+1},o_{t+2},\dots,o_T|i_t=q_i, \lambda)
>$$
>$\color{red} 关于\alpha 和\beta 这两个公式,  仔细看下, 细心理解.$
>
>观测序列概率$P(O|\lambda)$统一写成
>$$
>P(O|\lambda)=\sum_{i=1}^N\sum_{j=1}^N\alpha_t(i)a_{ij}b_j(o_{t+1}\beta_{t+1}(j)),\ t=1,2,\dots,T-1
>$$
>

### 概率与期望

概率

1. 输入模型$\lambda$与观测$O$, 输出在时刻$t$处于状态$q_i$的概率$\gamma_t(i)$
1. 输入模型$\lambda$与观测$O$, 输出在时刻$t$处于状态$q_i$且在时刻$t+1$处于状态$q_j$的概率$\xi_t(i,j)$
1. 在观测$O$下状态$i$出现的期望值
1. 在观测$O$下状态$i$转移的期望值
1. 在观测$O$下状态$i$转移到状态$j$的期望值

## 例子

### 例10.1

这个例子主要说明怎么从数据中拿到状态集合, 观测集合, 序列长度以及模型三要素.

分清楚哪些是已知, 哪些是推导得到.



### 例10.2

这个例子就是递归的矩阵乘法



### 例10.3

求最优状态序列



## 实际问题

### 手写数字生成

采样应用

隐马尔可夫模型的一个强大的性质是他对与时间轴上的局部的变形具有某种程度的不变性.

### 中文分词



###

## 参考

1. [^1]: [数学之美-CH05隐含马尔可夫模型, 吴军]()
1. [Wikipedia: Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
1. [hhmlearn](https://hmmlearn.readthedocs.io/en/latest/tutorial.html)

