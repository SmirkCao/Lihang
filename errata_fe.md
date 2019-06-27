# ERRATA
![Hits](https://www.smirkcao.info/hit_gits/Lihang/errata_fe.md)

参考书版本为**2017年11月第20次印刷**, 在这之后的印刷版本有可能进行过修订, 愿本书越来越完善.
1. $P_{162}$ 高斯混合模型的英文表示: Gaussian misture model $\rightarrow$ Gaussian mixture model

1. $P_{201}$对数线~~形~~模型$\rightarrow$对数线性模型

1. $P_{173}​$观测序列$O=\{红, 红, 白, 白, 红\}​$, 序列表示应该是$O=(红, 红, 白, 白, 红)​$

1. $P_{197}​$条件随机场(11.11)\~(11.12)， 应该是条件随机场(11.10)\~(11.11)， 这两个是线性链条件随机场模型的基本形式

1. $P_{198}​$公式(11.24)这个公式里面连乘用了行内形式，认为应该是行间形式，不算是错误了，书写上的一些问题。

1. $P_{200}$公式(11.30)中$M_i$应为$M_{i+1}$
   整体公式为$\beta_i(y_i|x)=[M_{i+1}(y_i,y_{i+1}|x)]\beta_{i+1}(y_{i+1}|x),i=1,2,\dots,n+1$

1. $P_{124}$中$b^*,f(x)$中的核函数表达式应该是$K(x_i,x_j)$以及$K(x,x_i)$

1. $P_{34}$算法2.2在模型输出以及步骤(3)中混用了$\sum_{j=1}^N$行间表达方式

1. $P_{75}$参考文献3这本书作者少写了一个Olshen

1. $P_{75}$参考文献7，ESL这本神书，在本书中的引文形式通常是有中译本说明的那种形式，应该统一一下。

1. $P_{53}$参考文献1，书中引用的是2005年的Draft，原链接更新了2017年的手稿，这部分内容变成了**Chapter 3**，补充下， 文件名更新了，新文件名是NBayesLogReg.pdf，差一个字母

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

28. $P_{189}$ 参考文献[2]的格式， 缺少卷数和页码范围， 77(2):257-186

29. $P_{12}$图1.2中的纵坐标， 应该是$y$，在PRML中误差函数是$E(w)=\frac{1}{2}\sum_{n=1}^N\{y(x_n-w)-t_n\}^2$所以纵坐标是$t$

30. $P_{57}​$ 在讲到决策树学习的损失函数部分。`决策树学习的损失函数通常是正则化的极大似然函数。决策树学习的策略是以损失函数为目标函数的最小化`这部分觉得描述有点问题，前面部分理解为正则化的似然函数作为损失函数，这个应该是对数似然，因为作为损失函数应该是越小越好，正则化的似然应该是越大越好。这样才能对应后面的`以损失函数为目标函数的最小化`

31. $P_{36}$参考文献2, 这个文献应该是On convergence proofs for perceptrons. repo里面参考文献下载脚本可以自动下载该文献， 是一份扫描档。 不过，有其他文献也按照本书的引用方法引用的。

32. $P_{134}$参考文献5, Platt这个文章最多引用的是J. Platt. *Advances in Kernel Methods -- Support Vector Learning,* *MIT Press,* *Cambridge, MA,* (*1998*)， 可以参考https://www.bibsonomy.org/bibtex/2ad411b41c7af4289282067a770edbdde/telekoma, 原书给的链接也是有效的，微软对这个链接做了转发， 跳转到新地址https://www.microsoft.com/en-us/research/publication/fast-training-of-support-vector-machines-using-sequential-minimal-optimization/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2F%3Fid%3D68391

33. $P_{36}$参考文献5, 现在比较容易获得的参考文献是1999年在Machine Learning上发表的那个版本，这个不算是错误。在repo的参考文献downloader里面，有对应的链接。

34. $P_{134}$参考文献1,没有标明页码，1995,20:273,297

35. $P_{XIII}$符号表说明中有关$||\cdot||_2$的说明， 是二范数，这个应该是对的。后面支持向量机部分$P_{114}$中描述支持向量机损失函数第二项$\lambda ||w||^2$为系数为$\lambda$的$w$的$L_2$范数，是正则化项。应该是二范数的平方。对应了$w\cdot w=||w||^2$， $w \cdot w$是在Vapnik的SVN文章中的表示方法。

36. $P_{122}$高斯核函数(Gaussian kernel  function)英文部分kernel和function之间，多了一个空格

37. $P_{118}$支持向量机部分，使用了核函数的分类决策函数拉格朗日乘子变成了$a$，求和范围变成了$N_s$，但是文中没有说明为什么做这种改变。

38. $P_{122}$介绍常用核函数的时候， 分类决策函数也用到了上面的表达方式。这两条， 涉及到的公式有7.68,7.89,7.90,7.91

39. $P_{122}$公式7.91分类决策函数中的$z$应该是$x_i$

    $f(x)=sign\left(\sum_{i=1}^{N_s}a_i^*y_i\exp\left(-\frac{||x_i-x||^2}{2\sigma^2}\right)+b^*\right)$

40. $P_{124}$称为**非线性支持向量**，应该是**非线性支持向量$\color{red}机$**。

41. $P_{182}$上面第一个公式，$\pi_{i_0}$应该是$\pi_{i_1}$

42. $P_{211}$这页表格中学习策略列，格式不是很统一，注意HMM部分`极大似然估计，`占了一行，而其他都是占了两行，应该是标点符号的空格处理不一样导致，不算错误，就是看起来稍微不同。

43. $P_{195}$图11.5,条件随机场是无向图模型，图中应该没有箭头。

44. $P_{199}$公式(11.26)中，$y$应该有下角标，$\alpha_0(y_0|x)$以及$y_0=start$

45. $P_{138}$算法8.1的描述中，$D_1$中用到了$w_{1i}$这样的表达，在后面$D_{m+1}$中用到了$w_{m+1,i}$这样的表达，意思完全明白，只是格式不太一致。

46. ~~$P_{138}$分类误差率这个定义里面，$e_m=\sum\limits_{i=1}\limits^NP(G_m(x_i)\neq y_i)$，这个应该没有求和符号$e_m=P(G_m(x_i)\neq y_i)$，在更新权重分布的时候做了归一化使得分类错误的点的系数求和应该刚好等于分类错误的概率。~~这个暂时保留下意见。

47. $P_{79}$描述$x_i\in \R^n$应该是$x_i\in \R^{n+1}$，下面用到的是扩充权重向量

48. $P_{89}$ `对数似然函数的极大值\hat{w}`应该是对数似然函数极大值对应的参数向量$\hat{w}$

49. $P_{93}$ 第三点约束最优化问题的第一条约束应该是期望$E_P(f_i)-E_{\tilde{P}}(f_i)=0,i=1,2,\cdots,n$

50. $P_{229}$ cell，47页没有cell相关的内容

