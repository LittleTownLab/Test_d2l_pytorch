# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %matplotlib inline
import random
import torch
from d2l import torch as d2l


# + pycharm={"name": "#%%\n"}
def sythetic_data(w, b, num_examples):
    """生成 y = Xw + b +噪声  """
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


# + pycharm={"name": "#%%\n"}
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = sythetic_data(true_w, true_b, 1000)

# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}
labels.size()
labels

# + pycharm={"name": "#%%\n"}
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


# + pycharm={"name": "#%%\n"}
def data_iter(batch_size, feature, labels):
    num_examples = len(feature)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield feature[batch_indices], labels[batch_indices]


# + pycharm={"name": "#%%\n"}
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X,'\n', y)
    break

# + pycharm={"name": "#%%\n"}
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# + pycharm={"name": "#%%\n"}
def linreg(X, w, b):
    """"线性回归模型"""
    return torch.matmul(X, w) + b


# + pycharm={"name": "#%%\n"}
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


# + pycharm={"name": "#%%\n"}
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()


# + pycharm={"name": "#%%\n"}
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for eporch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch B{eporch + 1}, loss {float(train_l.mean()):f}')

# + pycharm={"name": "#%%\n"}
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# + pycharm={"name": "#%%\n"}
printf("nothing")
