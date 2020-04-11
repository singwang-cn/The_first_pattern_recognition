'''
Created on 6 Mar 2020

@author: SingWang
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
#data processing
pima = pd.read_csv('diabetes.csv')
pima_tr = pima.sample(n = 200)
pima_te = pima.sample(n = 332)

print(pima_tr)

def recognization_line(x):
    return .25*x

x = np.arange(1, 200, 0.1)
y = recognization_line(x)

plt.figure('Original data')
plt.scatter(pima_tr['Glucose'], pima_tr['BMI'], marker = '.', c = pima_tr['Outcome'])
plt.xlabel('glu')
plt.ylabel('bmi')

plt.plot(x, y)
plt.show()


