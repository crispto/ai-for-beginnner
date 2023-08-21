from mxnet import gluon
from mxnet import autograd
from mxnet import init
import  matplotlib.pyplot as plt 
from mxnet.gluon import data as gdata


#采用 gluon 的方式来训练衣物分类模型的例子
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# X, y = mnist_train[:18]
# show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y))
batch_size=  200
totensor = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(dataset=mnist_train.transform_first(totensor), batch_size=batch_size, shuffle=False, num_workers=1)
test_iter = gdata.DataLoader(dataset=mnist_test.transform_first(totensor), batch_size=batch_size, shuffle=True, num_workers=1)



# 有 10 个类别，输出层的节点数应该也是10
net =gluon.nn.Sequential()
net.add(gluon.nn.Dense(256, activation='relu'), gluon.nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

loss = gluon.loss.SoftmaxCrossEntropyLoss()


trainer = gluon.Trainer(net.collect_params(), "sgd", {'learning_rate': 0.5})
import numpy as np 
    
def cal_accuracy(data_iter):
    pt,total = 0, 0
    for X, y in data_iter:
        ret = net(X)
        pred = np.argmax(ret , axis=1)
        pt += (pred.astype(y.dtype) == y).sum().asscalar()
        total += y.shape[0]
    # print(f"pt:{type(pt)}, total:{type(total)}")
    return float(pt/total)

def training(epoch = 5):
    for epoch in range(epoch):
        total_loss = 0.0
        iter_count = 0
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
                # print("type of ls is ", type(l), "shape of l is ", l.shape)
                total_loss += l.sum().asscalar() # l 是一个 batch_size 长度的向量，每一个代表一个样本的损失
                iter_count += l.shape[0]
            l.backward()
            trainer.step(batch_size)
        print(f"accurate in epoch {epoch} is {cal_accuracy(test_iter)}, training loss = {total_loss/iter_count}")

import argparse
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="训练模型或者展示数据集")
        # two subcommand : train and show
        subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
        # create the parser for the "train" command
        parser_train = subparsers.add_parser("train", help="train help")
        parser_train.add_argument("--epoch", type=int, default=5, help="epoch number")
        # create the parser for the "show" command
        parser_show = subparsers.add_parser("show", help="show help")
        parser_show.add_argument("--num", type=int, default=18, help="number of images to show")
        args = parser.parse_args()
        if args.subparser_name == "train":
            training(args.epoch)
        elif args.subparser_name == "show":
            X, y = mnist_train[:args.num]
            show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y))
            plt.show()
    