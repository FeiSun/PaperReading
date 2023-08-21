# Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System (KDD2018)

Jiaxi Tang, Ke Wang, 

School of Computing Science, Simon Fraser University, British Columbia, Canada

参考：Ranking Distillation 知识蒸馏应用在推荐系统中分享 - 积极废人的文章 - 知乎 https://zhuanlan.zhihu.com/p/362945179

这篇论文基于推荐系统，探索使用将知识蒸馏技术用在learning to rank问题上。RD和KD的不同地方在于，一般KD的蒸馏是用在分类任务上，标签类别有限，而RD的蒸馏应用在ranking任务上，而ranking的文档或者item量级很大，不能直接将KD这种的方法应用。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403134922095.png" alt="drawing" width="500"/>

## 任务

排序任务ranking目的是给定一个query或者user，来对document或者item进行排序，其中文档或者item量级一般较大。训练数据一般为人工标注一部分文档和query的相关性，可以是二分类（相关/不相关），或者是（强相关/相关/不相关），一般排序模型在这些有标签的数据上进行训练。进而来计算query 和文档的相似分数，来优化模型参数。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403135340525.png" alt="drawing" width="300"/>

排序的损失可以分为point-wise、pair-wise、list-wise，这篇论文主要是基于point-wise进行研究展开的。其中经典的point-wise损失：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403140311593.png" alt="drawing" width="500"/>

$ y_{d+} $ 是相关的文档， $ y_{d-} $ 是不相关的文档。

pair-wise损失：

![image-20230403152113283](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403152113283.png)

一般从两个方面来提升模型排序的效果，一个是在数据不会发生过拟合情况下，提升模型大小，来获得复杂的query和文档的交互隐式信息。第二个就是提供更多的训练数据。但是第一种往往伴随着牺牲了效率，第二种相对来说获得有标签数据较难。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403140925480.png" alt="drawing" width="500"/>

## 模型

这篇论文的蒸馏核心思想就是，首先分为两部分数据，一部分数据是带有标签的数据，另一部分是没有标签的数据。

训练过程：

1. 即首先用有标签数据训练一个较好的教师模型，然后拿无标签数据用教师模型进行预测，返回Top K靠前的文档。
2. 学生模型的训练损失分为两部分，一部分是在有标签数据下计算的损失，另一部分是对这Top K没有标签的数据进行预测出分数，来计算蒸馏损失。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403141452680.png" alt="drawing" width="700"/>

由于教师模型训练较好，从无标签的数据中预测的Top K个文档都是和query关联较大的，所以被看作正样本。这样的蒸馏方法就可以让学生模型既从有标签数据中进行学习，还可以从无标签中数据来学习额外知识。

学生模型具体Loss如下：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403141619803.png" alt="drawing" width="500"/>

把老师模型输出的Top K个文档看作正样本，计算损失如下：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403141754331.png" alt="drawing" width="500"/>

其中 $w_r$ 是针对Top K中每个item的权重，一般有两种直接计算权重的方法，第一种是按顺序衰减，位置越靠前的影响即越大， $w_r=1/r$，$ r $代表文档中排序的位置；还有一种是所有文档的权重都一样， $w_{r}=1 / K$ 。此外，这篇论文提出了三种新的计算权重的方法。($w^a_r\varpropto e^{-r/ \lambda}$，动态权重$w^b_r$，混合权重)

## 数据集

数据中包含大量的顺序信号，因此适合于顺序推荐。

![image-20230403142837377](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403142837377.png)

## 实验

选取的baseline模型也是序列模型(Fossil 和 Caser)。评价标准是Precision精确率，nDCG和MAP。

T：使用标签数据训练的老师模型

RD：学生模型采用自己与真实标签loss和蒸馏loss

S：是只采用与真实标签的loss。

![image-20230403142956002](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403142956002.png)

有些的实验结果经过RD蒸馏后的效果优于教师模型。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230403143604352.png" width="500"/>

经过蒸馏后的模型参数能降低一倍，推理时间也能加速近一倍。



这篇论文在蒸馏过程中，只使用了教师模型的输出结果，相当于黑盒。这篇论文将蒸馏较早的尝试应用在排序推荐中，模型提升可能来自两方面：模型蒸馏，增加的训练数据。

