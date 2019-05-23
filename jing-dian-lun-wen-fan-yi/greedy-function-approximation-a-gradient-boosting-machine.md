## Greedy Function Approximation: A Gradient Boosting Machine

### Abstract

函数的预估和近似往往是从函数空间，而不是从参数空间的数值优化来看待。 可以看到stagewise additive expansions (阶段增量式扩展)和steepest-descent minimization (最陡峭下降最小化)是有关联的。 我们提出了一种通用的梯度下降boosting（提升）范式来解决求和式扩展，这种范式可以基于任何拟合函数。 具体来说，对回归的话有最小方差，最小绝对差，以及Huber-M损失，对分类来说有多分类的logistic likelihood\(逻辑似然函数\)。 我们还使用回归树作为基本的增量组件，_作为一个特别的例子，并且带来特殊的增强_。并且我们还给出了这种TreeBoost模型的解释工具。 基于回归树的梯度提升法对回归和分类问题都能带来有竞争力，鲁棒性强并且可解释的过程，尤其适合处理有噪音的数据。 我们也讨论了这个方法和其他boosting方法（Freund and Shapire 1996, Friedman, Hasti and Tibsirani 2000）的关系。

### 1 Function estimation 函数估计

在函数估计或者说预测学习的问题里，一般会有一个随机的输出变量$y$, 以及一系列的“输入”或“解释因子”的随机变量 $X=\{x_1,...x_n\}$ 使用已知 $(y,X)$ 值的一组训练样本 ${y_i, X_i}$ , 目标是获得一个 $F(X)$ 的预估或近似 $\tilde F(x)$，等同于最小化在所有 $(y,X)$ 值的联合分布上，一个特定的损失函数$L(y, F(x))$ 的期望：

$$ F^{*} = argminE_{y,X}L(y,F(x)) = argminE_X[E_y(L(y,F(X)))|X] $$

常用的损失函数 $L(y,F)$ 包括，对回归有平方误差 $(y-F)^2$ 和绝对误差 $|y-F|$，其中 $y \in R^1$, 而对分类则有，负二项对数似然， $log(1+e^{-2yF})$，其中 $y \in \{-1,1\}$。

一个常见的方式是把 $F(X)$ 定义成一个参数化的函数族的成员 $F(X;P)$ ，其中 $P=\{P_1,P_2,...\}$ 是一个参数的有限集合，他们的所有值决定了一个函数成员。本文集中讨论增量式扩展型函数，形如：
$$ F(X;\{\beta_m, a_m\}_1^M)=\sum_{m=1}^M\beta_mh(X;a_m) $$

这个函数 $h(X;a)$ 是由参数 $a=\{a_1,a_2,...\}$ 所定义的关于输入变量 $X$ 的参数化函数。 The individual terms differs in the joint values am chosen for these parameters. 这样的扩展法是许多函数近似方法的核心，比如神经网络，radial basis functions, MARS, wavelets，以及支持向量机。 本文关注的是当 $h(X;a_m)$是一个小回归树如CART的情况。对一棵回归树来说，参数 $a_{m}$ 指的是分支变量，分支位置以及每棵树的叶子节点的均值。

#### 1.1 数值优化

一般而言，选择一个参数化模型 $F(X;P)$ 就把函数最优问题变成一个参数最优化问题

 $$P^*=argmin_PO(P)$$
 $$O(P)=E_{y,X}L(y,F(X;P))$$

 则 $F^*(X)=F(x;P^*)$

 对大多数的F(X;P)和L, 一定要用数值优化来解决上面的问题。我们通常会把参数的解法写成以下的形式：

 $$P^*=\sum_{m=0}^Mp_m$$

 其中 $p_0$ 是一个初始的猜测，而 $\{p_m\}_1^M$是连续的改进（步进或者提升），每一次提升都基于之前的步进。而计算每一个步进的方法由最优化方法来定义。

#### 1.2 Steepest-descent 最陡峭下降
