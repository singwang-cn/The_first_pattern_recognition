'''
Created on 20 Feb 2020

@author: SingWang
'''

import matplotlib.pyplot as plt
from sklearn import datasets
import torch


iris_data = datasets.load_iris()

x = torch.tensor([iris_data.data[:, 2],
                  iris_data.data[:, 3]]).float()

n = x.size()[1]

species = iris_data.target

u = torch.tensor([[x[0].mean()],
                  [x[1].mean()]])
sig = (x-u).mm((x-u).t()) / n

#s = torch.tensor([[0.92, 0.39],
#                  [0.39, -0.92]])

x_regula_0 = (x - u)[0] / (sig[0, 0] ** .5)
x_regula_1 = (x - u)[1] / (sig[1, 1] ** .5)

s = sig.eig(eigenvectors = True)[1]
#print(s)

lam = s.inverse().mm(sig.mm(s))
#print(lam)

x_deco = s.inverse().mm(x)


plt.figure('Before regularization')
plt.scatter(x[0], x[1], c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.figure('After regularization')
plt.scatter(x_regula_0, x_regula_1, c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.figure('After decorrelation')
plt.scatter(x_deco[0], x_deco[1], c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.show()
