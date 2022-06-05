# CH24 卷积神经网络

从数学卷积开始，然后介绍二维卷积，填充和步幅，三维卷积。

关于数学卷积，在scipy.signal里面有。

另外还有个例子，方波变三角波的那个。

数学卷积是**定义**在两个函数上的运算，**表示**其中一个函数对另一个函数的形状上进行的调整。

书中提了一句，数学卷积可以自然的拓展到二维和离散的情况。注意在离散的情况下，输入x和核函数w都是数组。scipy.convolve已经抛弃了，numpy.convolve为替代方案，numpy.convolve里面有离散的形式的定义。
$$
(a * v)[n] = \sum_{m = -\infty}^{\infty} a[m] v[n - m]
$$

就书上的图24.1，有几个特点：

1. 顶点基本上重合了
2. 幅度一样
3. 看起来和插值有点像

## 参考

1. [Wikipedia Convolution词条](https://en.wikipedia.org/wiki/Convolution)
2. [深入理解卷积](https://blog.csdn.net/weixin_37682263/article/details/87914913)
