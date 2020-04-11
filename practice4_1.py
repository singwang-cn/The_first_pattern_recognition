'''
Created on 17 Feb 2020

@author: SingWang
'''

import matplotlib.pyplot as plt
from sklearn import datasets
import torch

iris_data = datasets.load_iris()
#print(iris_data)
x = torch.tensor([iris_data.data[:, 2],
                  iris_data.data[:, 3]])

n = x.size()[1]

species = iris_data.target

#u = torch.tensor([[3.76], [1.20]])
u = torch.tensor([[x[0].mean()],
                  [x[1].mean()]])
#print(u)

#sig = torch.tensor([[3.12, 1.30],
#                    [1.30, 0.58]])
sig = (x-u).mm((x-u).t()) / n
#print(sig)

x_regula_0 = (x - u)[0] / (sig[0, 0] ** .5)
x_regula_1 = (x - u)[1] / (sig[1, 1] ** .5)


plt.figure('Before regularization')
plt.scatter(x[0], x[1], c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.figure('After regularization')
plt.scatter(x_regula_0, x_regula_1, c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.show()
