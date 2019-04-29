# Greedy Function Approximation: A Gradient Boosting Machine

### Greedy Function Approximation: A Gradient Boosting Machine

#### Abstract

函数的预估和近似往往是从函数空间，而不是从参数空间的数值优化来看待。 可以看到stagewise additive expansions\(阶段增量式扩展\)和steepest-descent minimization\(最陡峭下降最小化\)是有关联的。 我们提出了一种通用的梯度下降boosting（提升）范式来解决求和式扩展，这种范式可以基于任何拟合函数。 具体来说，对回归的话有最小方差，最小绝对差，以及Huber-M损失，对分类来说有多分类的logistic likelihood\(逻辑似然函数\)。 我们还使用回归树作为基本的增量组件，_作为一个特别的例子，并且带来特殊的增强_。并且我们还给出了这种TreeBoost模型的解释工具。 基于回归树的梯度提升法对回归和分类问题都能带来有竞争力，鲁棒性强并且可解释的过程，尤其适合处理有噪音的数据。 我们也讨论了这个方法和其他boosting方法（Freund and Shapire 1996, Friedman, Hasti and Tibsirani 2000）的关系。

#### 1 Function estimation 函数估计

在函数估计或者说预测学习的问题里，一般会有一个随机的输出变量$y$, 以及一系列的“输入”或“解释因子”的随机变量 $X={x\_1,...x\_n}$ 使用已知 $\(y,X\)$ 值的一组训练样本 ${y\_i, X\_i}$ , 目标是获得一个 $F^\*\(X\)$ 的预估或近似 $\tilde F\(x\)$，

