'''
Created on 26 Feb 2020

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

x_regula_0 = (x - u)[0] / (sig[0, 0] ** .5)
x_regula_1 = (x - u)[1] / (sig[1, 1] ** .5)

s = sig.eig(eigenvectors = True)[1]

lam = s.inverse().mm(sig.mm(s))

x_whitened = (lam ** .5).inverse().mm(s.t().mm((x-u)))

print(x_whitened)

plt.figure('Before regularization')
plt.scatter(x[0], x[1], c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.figure('After whitening')
plt.scatter(x_whitened[0], x_whitened[1], c = species)
plt.xlabel('length of iris')
plt.ylabel('width of iris')

plt.show()