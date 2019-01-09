# ERRATA

参考书版本为**2017年11月第20次印刷**, 在这之后的印刷版本有可能进行过修订, 愿本书越来越完善.
1. $P_{162}$ 高斯混合模型的英文表示: Gaussian misture model $\rightarrow$ Gaussian mixture model

1. $P_{201}$对数线~~形~~模型$\rightarrow$对数线性模型

1. $P_{173}$观测序列$O={红, 红, 白, 白, 红}$, 序列表示应该是$O=(红, 红, 白, 白, 红)$

1. $P_{197}$条件随机场(11.11)\~(11.12)， 应该是条件随机场(11.10)\~(11.11)， 这两个是线性链条件随机场模型的基本形式

1. $P_{198}$公式(11.24)这个公式里面连乘用了行内形式，认为应该是行间形式，不算是错误了，书写上的一些问题。

1. $P_{200}$公式(11.30)中$M_i$应为$M_{i+1}$
   整体公式为$\beta_i(y_i|x)=[M_{i+1}(y_i,y_{i+1}|x)]\beta_{i+1}(y_{i+1}|x),i=1,2,\dots,n+1$

1. $P_{124}$中$b^*,f(x)$中的核函数表达式应该是$K(x_i,x_j)$以及$K(x,x_i)$

1. $P_{34}$算法2.2在模型输出以及步骤(3)中混用了$\sum_{j=1}^N$行间表达方式

1. $P_{75}$参考文献3这本书作者少写了一个Olshen

1. $P_{75}$参考文献7，ESL这本神书，在本书中的引文形式通常是有中译本说明的那种形式，应该统一一下。

1. $P_{53}$参考文献1，书中引用的是2005年的Draft，原链接更新了2017年的手稿，这部分内容变成了**Chapter 3**

1. $P_{47}$最后一段参数个数$K\prod_{j=1}^nS_j$书中混用了行间表达形式和行内表达形式

1. $P_{154}$参考文献9，这个文章是2002年的文献，书中记录为2004，这文章也不错

1. $P_{164}$公式9.29,第二个求和应该是对$j$求和,从取值范围到$N$应该也可以看出$\sum_{j=1}^N\hat\gamma_{jk}$

1. $P_{169}$ d维的形式应该是$j=3,4,\dots,d$而不是$j=3,4,\dots,k$

1. $P_{181}$`由于监督学习需要使用训练数据`这个应该是`需要使用标注的训练数据`.

1. $P_{230}$ 海赛矩阵 Hesse matrix, 应该是 Hessian Matrix

1. $P_{156}$观测数据表示为$Y=(Y_1, Y_2, Y_3, \dots, Y_n)^T$, 未观测数据表示为$Z=(Z_1,Z_2, Z_3,\dots, Z_n)^T$, 则观测数据的似然函数为

     > 其实觉得这里应该是小写的$y=(y_1,y_2,\dots,y_n), z=(z_1, z_2, \dots,z_n)$

1. $P_{219}$ Hesse matrix -> Hessian Matrix

1. $P_{80}$公式6.7， 关于多项逻辑斯谛回归模型中的求和部分下角标如果换成$i$，觉得更好理解一点
      $$
      \begin{aligned}
      P(Y=k|x)&=\frac{\exp(w_k\cdot x)}{1+\sum_{j=1}^{K-1}\exp(w_j\cdot x)}, k=1,2,\dots,K-1\\
      P(Y=k|x)&=\frac{1}{1+\sum_{j=1}^{K-1}\exp(w_j\cdot x)}\\
      \end{aligned}
      $$

1. $P_{153} , P_{146}$`提升树是以分类树或回归树为基本分类器的提升方法`这里面基本分类器应该是基函数，分类问题对应分类树， 回归问题对应回归树。

1. $P_{140}$例题来源于http://www.csie.edu.tw， 这个大概应该是http://www.csie.ntu.edu.tw。 但是也没找到对应的例子页面。

1. $P_{148}$在提升树这个地方， 最后得到的提升树是$f_M(x)$， 前面介绍加法模型的时候， 得到的是$f(x)$实际上是一样的意思， 但是两个地方的表达不太一样。这个， 其实不算吧。。

1. $P_{170}$Baum与Welch算法，后面HMM的描述中用的是Baum-Welch算法， 同一本书两个表达方式不统一。其实，这个也不是太重要。

1. $P_{159}$
$$
\begin{align}
L(\theta)-L(\theta^{(i)})&=\log \left(\sum_Z\color{green}P(Y|Z,\theta^{(i)})\color{black}\frac{P(Y|Z,\theta)P(Z|\theta)}{\color{green}P(Y|Z,\theta^{(i)})}\color{black}\right)-\log P(Y|\theta^{(i)})\\
&\ge\sum_Z P(Z|Y,\theta^{(i)})\log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}-\log P(Y|\theta^{(i)})\\
&=\sum_Z P(Z|Y,\theta^{(i)})\log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}-\color{red}\sum_ZP(Z|Y,\theta^{(i)})\color{black}\log P(Y|\theta^{(i)})\\
&=\sum_ZP(Z|Y,\theta^{(i)})\log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}
\end{align}
$$
这里绿色部分应该是$P(Z|Y,\theta^{(i)})$，为了构建期望而凑项，进而应用琴声不等式。

26. $P_{162}$ 关于定理9.2.2的证明，参阅文献[6]， 这个定理的证明应该在参考文献[5]中有提到。
27. $P_{166}$ `将其对  求偏导 ` 这个地方的符号$\widetilde{P}$， 应该是$\tilde{P}$， 和定义9.3中的有差异，一个是widetilde，一个是tilde，统一最好。
28. 

