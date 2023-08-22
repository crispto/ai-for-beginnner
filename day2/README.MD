# 基本的线性回归和 分类 模型的实现

linear_regression_scratch_demo 从零开始实现了简单的线性回归模型，参数、loss、网络都自己定义，只是使用了mxnet 的自动求导功能。
fashion_mnist_demo 是使用 mxnet 的高级接口实现的分类模型，训练了一个简单的衣服分类模型。

### BUG 记录 mxnet 和 numpy 的版本有个冲突，

![Alt text](image.png)

解决办法是将 numpy 降级到 1.23.1， 其中 np.bool 还在使用
