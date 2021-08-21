# BayesianCNN-Based-on-Paddle
## 概述

复现论文地址：[Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf)

传统神经网络存在两个主要问题：容易过拟合、对预测结果过自信

引入贝叶斯的概念在神经网络中可以解决以上问题：

- 将权重作为随机变量看待，不易过拟合。贝叶斯神经网络在小型数据集上也能很好的学习. 先验的加入相当于给网络提供了一种约束和正则，Dropout 在分析中也被认为是贝叶斯神经网络的一种形式。
- 贝叶斯神经网络能够产生不确定性的度量，而非仅给出一个判别结果。

## 复现环境

paddle == 2.1.0

Teslav100 16GB

## 复现要求

MNIST手写数据集的Test数据集上，达到Test error **1.32%**

## AI Studio地址

https://aistudio.baidu.com/aistudio/projectdetail/2301492

## 结果

|   模型    |                  Test err（复现目标1.32%）                   |
| :-------: | :----------------------------------------------------------: |
| LeNet-BBB | **1.25%**（由于正态采样部分是随机生成的，最终跑出的test err不确定，最低可达到1.20%甚至更低） |
|   LeNet   |                            1.56%                             |

## 训练命令

```sh
python train.py
```

## 测试命令

快速测试（注意修改模型的路径）

```
 python test.py
```



