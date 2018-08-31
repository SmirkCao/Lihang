# CH6 逻辑斯谛回归与最大熵模型

## 前言

<统计学习方法>的本章结构

1. 逻辑斯谛回归模型
   1. 逻辑斯谛分布
   1. 二项逻辑斯谛回归模型
   1. 模型参数估计
   1. 多项逻辑斯蒂回归**模型**
1. 最大熵模型
   1. 最大熵原理
   1. 最大熵模型定义
   1. 最大熵模型学习
   1. 极大似然估计
1. 模型学习的最优化算法
   1. 改进的迭代尺度法
   1. 拟牛顿法

## 模型



### 逻辑斯谛回归模型

在<机器学习>上把这个叫做对数几率回归





### 最大熵模型

#### 概念

逻辑斯谛回归模型和最大熵模型, 既可以看作是概率模型, 又可以看作是非概率模型. 

##### 信息量

信息量是对信息的度量, PRML中有关于信息量的讨论,  信息是概率的单调函数.

$h(x)=-\log_2{p(x)}$, 符号保证了非负性. 低概率事件对应了高的信息量. 对数底选择是任意的, 信息论里面常用2, 单位是比特.

- 信息和概率的关系参考PRML中1.6节信息论部分的描述.

> 如果我们知道某件事件一定会发生, 那么我们就不会接收到信息.
> 于是, 我们对于信息内容的度量将依赖于概率分布$p(x)$
>
> 如果我们有两个不相关的事件x, y, 那么我们观察到两个事件同时发生时获得的信息应该等于观察到事件各自发生时获得的信息之和, 即$h(x,y)=h(x)+h(y)$,  这两个不相关的事件是独立的, 因此$p(x,y)=p(x)p(y)$
>
> **根据这两个关系**, 很容易看出$h(x)$一定与$p(x)$的对数有关. 所以有
>
> $$h(x)=-\log_2{p(x)}=\log_2{\frac{1}{p(x)}}$$
>
> - 负号确保了信息非负
> - 低概率事件$x$对应了高的信息.



##### 熵和概率

熵可以从随机变量状态需要的平均信息量角度理解, 也可以从描述统计力学中无序程度的度量角度理解.

关于熵, 条件熵, 互信息, 这些内容在[第五章](../CH5/README.md)5.2节有对应的描述.

> 下面看下信息熵在PRML中的表达
>
> 假设一个发送者想传输一个随机变量$x$的值给接受者. 在这个过程中, 他们传输的平均信息量可以通过求**信息$h(x)$关于概率分布$p(x)$的期望**得到.
>
> 这个重要的量叫做随机变量$x$的熵

**Venn图**辅助理解和记忆, 这个暂时不画, 下面考虑下为什么Venn图能帮助理解和记忆?

因为熵的定义把连乘变成了求和, 对数的贡献. 这样可以通过集合的交并来实现熵之间关系的理解.

1. **概率** $\sum _{i=1}^{n}{p_i=1}$ $p \in [0,1]$

1. **熵**$Ent(D) \in [0, \log_2{|\mathcal Y|}]$, 熵可以大于1. 熵是传输一个随机变量状态值所需的比特位下界(信息论角度的理解)

1. **信息熵**是度量样本集合纯度最常用的一种指标.

   $Ent(D)=-\sum \limits ^{|\mathcal Y|}_{k=1}p_k\log_2{p_k}$

   - if $p=0$, then $p\log_2{p}=0$
   - $Ent(D)$越小, D的纯度越高. 非均匀分布比均匀分布熵要小.
   - 熵衡量的是不确定性, 概率描述的是确定性

1. **联合熵(相当于并集)**

   $H(X, Y) = H(X) + H(Y|X) = H(Y)+H(X|Y) = H(X|Y)+H(Y|X)+I(X;Y)$

   这个通过Venn应该是相对容易记忆, 是不是容易理解这个又汉姆雷特了.

   如果X和Y独立同分布, 联合概率分布$P(X,Y)=P(X)P(Y)$ 

1. **条件熵**

   条件熵是最大熵原理提出的基础,最大的是条件熵, 这个在书中有写(定义6.3).

   条件熵衡量了条件概率分布的均匀性.

   最大熵, 就是最大这个条件熵

   find 

    $$\begin{aligned}
   p^*&=\arg\max\limits_{p\in \mathcal C}H(p)\\
   &=\arg \max\limits_{p\in \mathcal C}(-\sum\limits_{x,y} {\tilde p(x)p(y|x)\log p(y|x) })
   \end{aligned} $$

   接下来的概念, 把熵的思想应用在模式识别问题中. 

1. **互信息**

   互信息(mutual information), 对应熵里面的交集, 常用来描述差异性

   一般的, 熵$H(Y)$与条件熵$H(Y|X)$之差称为互信息.  注意一下, 这里[第五章](../CH5/README.md)中用到了$H(D, A)$ 可以对应理解下.

   1. feature selection 
   1. Feature Correlation, 刻画的是相互之间的关系. 相关性主要刻画线性, 互信息刻画非线性

1. **信息增益**

   这个对应的是第五章的内容, 决策树学习应用信息增益准则选择特征.
   $$
   g(D,A)=H(D)-H(D|A)
   $$
   信息增益表示得知X的信息而使类Y的信息的不确定性减少的程度.

   在决策树学习中, 信息增益等价于训练数据集中类与特征的互信息.

1. **相对熵 (KL 散度)** 

   相对熵(Relative Entropy)描述差异性, 从分布的角度描述差异性, 可用于度量两个概率分布之间的差异.

   KL散度不是一个度量.

   KL散度满足非负性.

   > 考虑由$p(x,y)$给出的两个变量x和y组成的数据集. 如果变量的集合是独立的, 那么他们的联合分布可以分解为边缘分布的乘积$p(x,y)=p(x)p(y)$
   >
   > 如果变量不是独立的, 那么我们可以通过考察**联合分布**与**边缘分布乘积**之间的KL散度来判断他们是否"接近"于相互独立.f
   >
   > $$I(x,y)=KL(p(x,y)|p(x)p(y))=-\iint p(x,y) \ln {\left( \frac{p(x)p(y)}{p(x,y)}\right)}$$
   >
   > 这被称为变量x和变量y之间的互信息.
   >
   > --PRML 1.6.1

   注意这里, 参考下[第五章](../CH5/README.md)中关于互信息的描述

   > 决策树学习中的信息增益等价于训练数据集中**类**与**特征**的互信息

   注意这里面类Y, 特征X.

   互信息和条件熵之间的关系
   $$
   I(x,y)=H(X)-H(x|y)=H(y)-H(y|x)
   $$
   可以把互信息看成由于知道y值而造成的x的不确定性的减小(反之亦然). *这个就是信息增益那部分的解释.*

1. **交叉熵**

   刻画两个分布之间的差异
   $$
   \begin{aligned}
   CH(p,q)&=-\sum\limits_{i=1}^{n}p(x_i)\log{q(x_i)}\\
   &=-\sum\limits_{i=1}^{n}p(x_i)\log{p(x_i)}+\sum\limits_{i=1}^{n}p(x_i)\log{p(x_i)}-\sum\limits_{i=1}^{n}p(x_i)\log{q(x_i)}\\
   &=H(p)+\sum\limits_{i=1}^{n}p(x_i)\log{\frac{p(x_i)}{q(x_i)}}\\
   &=H(p)+KL(p||q)
   \end{aligned}
   $$



   CNN时候常用

对于各种熵的理解, 是构建后面的目标函数的基础.

##### 最大熵原理

最大熵原理(Maxent principle)是**概率模型**学习的一个准则.

书中通过一个例子来介绍最大熵原理, 下面引用一下文献中关于这个例子的总结.

> Model all that is known and assume nothing about that which is unknown. In other words, given a collection of facts, choose a model which is consistent with all the facts, but otherwise as uniform as possible.
>
> -- Berger, 1996

书中关于这部分的总结如下：**满足约束条件下求等概率的方法估计概率分布**

关于最大熵原理有很多直观容易理解的解释, 比如Berger的例子, 比如吴军老师数学之美中的例子. 最大熵原理很常**见**, 很多原理我们都一直在用, 只是没有上升到理论的高度.

等概率表示了对事实的无知, 因为没有更多的信息, 这种判断是合理的.

最大熵原理认为要选择的概率模型首先必须满足**已有的事实**, 即**约束条件**

最大熵原理根据已有的信息（**约束条件**）, 选择适当的概率模型.

最大熵原理认为不确定的部分都是等可能的, 通过熵的最大化来表示**等可能性**.

最大熵的原则, 承认已有的, 且对未来无偏

最大熵原理并不直接关心特征选择, 但是特征选择是非常重要的, 因为约束可能是成千上万的.

##### 最大熵原理几何解释

这部分书中只描述了模型空间$\mathcal P$, 两个约束$C_1$和$C_2$是**一致性**约束的情况.

在Berger 1996里面有展开这部分,  分了四个图, 分别讨论了

1. 概率模型空间$\mathcal {P}$
1. 单一约束$C_1$
1. 一致性(consistent)约束$C_1$和$C_2$, 这种情况下模型唯一确定$p=C_1\bigcap C_2$
1. 非一致性(inconsistent)约束$C_1$和$C_3$, 这种情况下没有满足约束条件的模型.

##### 特征与约束条件

关于特征和约束, Berger有他的阐述

> 指示函数
>
> $f(x,y)=\begin{cases}
> 1 & if\ y=en\ and\ April\ follows\ in\\
> 0 & otherwise
> \end{cases}$

上面这个$f$直接引用自Berger的说明, 原来的例子是英语in到法语的翻译.

这里面f就是**特征函数**, 或者**特征**.

定义一个期望, 如果是二值函数的话, 就相当于计数. 通过样本得到的这个统计. 但是样本是有限的, 并不是一个真实的分布, 所以叫经验分布, 如果我们拿到的这个模型能够表示实际的分布, 那么就可以假设经验分布和真实分布是相等的. 这个, 就是**约束方程**, 或者**约束**.

一般模型的特征是关于x的函数, 最大熵模型中的特征函数, 是关于x和y的函数. 注意理解$f(x)$与$f(x, y)$的区别.

##### 模型

假设分类模型是一个条件概率分布, $P(Y|X)$, $X\in \mathcal {X} \sube \mathbf R^n$

给定一个训练集 $T=\{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$

N是训练样本容量, $x \in \mathbf R^n$ 

联合分布P(X, Y)与边缘分布P(X)的经验分布分别为$\widetilde P(X, Y)和\widetilde P(X)$

$$
\begin{aligned}
&\widetilde P (X=x, Y=y)=\frac{\nu(X=x, Y=y)}{N} \\
&\widetilde P (X=x)=\frac {\nu (X=x)}{N}
\end{aligned}
$$
上面两个就是不同的数据样本, 在训练数据集中的比例.

如果增加n个**特征函数**, 就可以增加n个**约束条件**

假设满足所有约束条件的模型集合为

$\mathcal {C} \equiv \ \{P \in \mathcal {P}|E_P(f_i)=E_{\widetilde {P}}(f_i) {, i=1,2,\dots,n}\} $

定义在条件概率分布$P(Y|X)$上的条件熵为

$H(P)=-\sum \limits _{x, y} \widetilde {P}(x)P(y|x)\log {P(y|x)}$

则模型集合$\cal {C}$中条件熵$H(P)$最大的模型称为最大熵模型, 上式中对数为自然对数.



特征函数$f(x,y)$关于经验分布$\widetilde P (X, Y)$的期望值, 用$E_{\widetilde P}(f)$表示

$$E_{\widetilde P}(f)=\sum\limits_{x,y}\widetilde P(x,y)f(x,y)$$

特征函数$f(x,y)$关于模型$P(Y|X)$与经验分布$\widetilde P (X)$的期望值, 用$E_{P}(f)$表示

$$E_{P}(f)=\sum\limits_{x,y}{\widetilde P(x)P(y|x)f(x,y)}$$

如果模型能够获取训练数据中的信息, 那么就可以假设这两个期望值相等, 即

$$E_P(f)=E_{\widetilde P}(f)$$

上面这个也是约束方程

#### 算法实现

##### 特征提取原理

通过对已知训练集数据的分析, 能够拿到联合分布的经验分布和边缘分布的经验分布.

特征函数用来描述$f(x, y)$描述输入x和输出y之间的某一事实.
$$
f(x,y) = \begin{cases}
1 & x与y满足某一事实\\
0 & 否则
\end{cases}
$$

这里, 满足的事实, 可以是in, 显然, 特征函数可以自己定义, 可以定义多个, ~~~这些就是约束 ~~~

之前理解的不对,  看前面有描述特征和约束的关系.

##### 预测分类原理
这里面重复一下书中的过程, 在$L(P, w)$对$P$求导并令其为零的情况下解方程能拿到下面公式
$$
P(y|x)=\exp{\left(\sum_{i=1}^{n}w_if_i(x,y)+w_0-1\right)}=\frac{\exp{\left(\sum\limits_{i=1}^{n}w_if_i(x,y)\right)}}{\exp{\left(1-w_0\right)}}
$$
书中有提到因为$\sum\limits{_y}P(y|x)=1$, 然后得到模型

$$
P_w(y|x)=\frac{1}{Z_w(x)}\exp{\sum\limits_{i=1}^{n}w_if_i(x,y)}\\
Z_w(x)=\sum_y\exp{\sum_{i=1}^{n}w_if_i(x,y)}
$$

注意这里面$Z_w$是归一化因子.

这里面并不是因为概率为1推导出了$Z_w$的表达式, 而是因为$Z_w$的位置在分母, 然后对应位置$\exp(1-w_0)$也在分母, 凑出来这样一个表达式, 意思就是遍历y的所有取值, 求分子表达式的占比.

#### 最大熵模型的学习

最大熵模型的学习过程就是求解最大熵模型的过程.

最大熵模型的学习可以形式化为约束最优化问题.
$$
\begin{eqnarray*}
\min \limits_{P\in \mathcal {C}}-H(P)=\sum\limits_{x,y}\widetilde P(x)P(y|x)\log P(y|x)\tag{6.14}\\
s.t. E_P(f_i)-E_{\widetilde P}(f_i)=0, i =1,2,\dots,n\tag{6.15}\\
\sum \limits_y P(y|x)=1\tag{6.16}
\end{eqnarray*}
$$

可以通过例6.2 来理解最大熵模型学习的过程, 6.2 考虑了两种约束条件, 这部分内容可以通过python符号推导实现, 西面代码整理整个求解过程.

##### 例6.2

###### 一个约束条件

```python
from sympy import *

# 1 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2", real=True)
L = P1 * log(P1) + P2 * log(P2) + P3 * log(P3) + P4 * log(P4) + P5 * log(P5) \
	+ w0 * (P1 + P2 + P3 + P4 + P5 - 1)
P1_e = (solve(diff(L, P1), P1))[0]
P2_e = (solve(diff(L, P2), P2))[0]
P3_e = (solve(diff(L, P3), P3))[0]
P4_e = (solve(diff(L, P4), P4))[0]
P5_e = (solve(diff(L, P5), P5))[0]
L = L.subs({P1: P1_e, P2: P2_e, P3: P3_e, P4: P4_e, P5: P5_e})
w = (solve([diff(L, w0)], [w0]))[0]
P = [P1_e.subs({w0: w[0]}),
     P2_e.subs({w0: w[0]}),
     P3_e.subs({w0: w[0]}),
     P4_e.subs({w0: w[0]}),
     P5_e.subs({w0: w[0]})]
P
```
###### 两个约束条件
```python
# 2 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2",real=True)
L = P1*log(P1) + P2*log(P2)+P3*log(P3)+P4*log(P4)+P5*log(P5)\
    +w1*(P1+P2-3/10)\
    +w0*(P1+P2+P3+P4+P5-1)
P1_e = (solve(diff(L,P1),P1))[0]
P2_e = (solve(diff(L,P2),P2))[0]
P3_e = (solve(diff(L,P3),P3))[0]
P4_e = (solve(diff(L,P4),P4))[0]
P5_e = (solve(diff(L,P5),P5))[0]
L = L.subs({P1:P1_e, P2:P2_e, P3:P3_e, P4:P4_e, P5:P5_e})
w = (solve([diff(L,w1),diff(L,w0)],[w0,w1]))[0]
P = [P1_e.subs({w0:w[0], w1:w[1]}),
     P2_e.subs({w0:w[0], w1:w[1]}),
     P3_e.subs({w0:w[0], w1:w[1]}),
     P4_e.subs({w0:w[0], w1:w[1]}),
     P5_e.subs({w0:w[0], w1:w[1]})]
P
```

###### 三个约束条件

```python
# 3 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2",real=True)
L = P1*log(P1) + P2*log(P2)+P3*log(P3)+P4*log(P4)+P5*log(P5)\
    +w2*(P1+P3-1/2)\
    +w1*(P1+P2-3/10)\
    +w0*(P1+P2+P3+P4+P5-1)
P1_e = (solve(diff(L,P1),P1))[0]
P2_e = (solve(diff(L,P2),P2))[0]
P3_e = (solve(diff(L,P3),P3))[0]
P4_e = (solve(diff(L,P4),P4))[0]
P5_e = (solve(diff(L,P5),P5))[0]
L = L.subs({P1:P1_e, P2:P2_e, P3:P3_e, P4:P4_e, P5:P5_e})
w = (solve([diff(L,w2),diff(L,w1),diff(L,w0)],[w0,w1,w2]))[0]
P = [P1_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P2_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P3_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P4_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P5_e.subs({w0:w[0], w1:w[1],w2:w[2]})]
P
```

## 模型学习

逻辑斯谛回归模型和最大熵模型学习归结为以**似然函数**为**目标函数**的最优化问题, 通常通过迭代算法求解.

### 目标函数



#### 逻辑斯谛回归模型

$$
\begin{aligned}
L(w)&=\sum\limits^{N}_{i=1}[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i\log{\frac{\pi(x_i)}{1-\pi(x_i)}}+\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i(w\cdot x_i)-\log(1+\exp(w\cdot{x_i})]
\end{aligned}
$$



#### 最大熵模型

$$
\begin{align}
L_{\widetilde {P}}(P_w)&=\sum \limits_{x,y}\widetilde {P}(x,y)\log{P}(y|x)\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x,y)\log{(Z_w(x))}\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}
\end{align}
$$

1. 逻辑斯谛回归模型与朴素贝叶斯的关系
1. 逻辑斯谛回归模型与AdaBoost的关系
1. 逻辑斯谛回归模型与核函数的关系

### 其他

课后习题的第一个题目提到了指数族(Exponential family)分布, 这个概念在PRML中有单独的章节进行阐述.

## 代码实现

关于代码实现, 网上看似众多的版本,应该基本上都源自最早15年的一份GIS的程序. 

无论怎样,这些代码的实现, 都会有助于对Maxent的理解.推荐后面参考文献[1]

李航老师在本章的参考文献头两位给的就是Berger的文章

## 参考

1. [Berger,1995, A Brief Maxent Tutorial](https://www.cs.cmu.edu/afs/cs/user/aberger/www/html/tutorial/tutorial.html)
1. [数学之美:信息的度量和作用]
1. [数学之美:不要把鸡蛋放在一个篮子里 谈谈最大熵模型]
1. [李航·统计学习方法笔记·第6章 logistic regression与最大熵模型（2）·最大熵模型](https://blog.csdn.net/tina_ttl/article/details/53542004)
1. [最大熵模型与GIS ,IIS算法](https://blog.csdn.net/u014688145/article/details/55003910)
1. [关于最大熵模型的严重困惑：为什么没有解析解？](https://www.zhihu.com/question/49139674/answer/114670380)
1. [最大熵模型介绍](http://www.cnblogs.com/hexinuaa/p/3353479.html) 这个是Berger的文章的翻译.
1. [理论简介](https://vimsky.com/article/714.html)  [代码实现](https://vimsky.com/article/776.html) 
1. [另外一份代码](https://github.com/WenDesi/lihang_book_algorithm/tree/master/maxENT)
1. [如何理解最大熵模型里面的特征？](https://www.zhihu.com/question/24094554)


