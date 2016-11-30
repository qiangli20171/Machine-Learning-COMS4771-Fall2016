"""
Python Code for HW5 Problem3
Neural Networks
Created on Wed Nov 22 2016
@author: Yizhou Wang
"""

import numpy as np
from scipy.io import loadmat

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import ElasticNet

housing = loadmat('housing.mat')
data = housing['data'].astype(float)
labels = housing['labels'].astype(float)
(ntr, d) = data.shape
testdata = housing['testdata'].astype(float)
testlabels = housing['testlabels'].astype(float)
(nte, d) = testdata.shape
'''
reg = LinearRegression()
reg.fit(data, labels)
preds = reg.predict(testdata)
ols_ls = mean_squared_error(testlabels, preds)
print 'OLS_LS =', ols_ls
'''
'''
reg = Lasso(alpha = 2.04)
reg.fit(data, labels)
print reg.coef_
preds = reg.predict(testdata)
lasso_ls = mean_squared_error(testlabels, preds)
print 'Lasso_LS =', lasso_ls
'''
'''
reg = LassoLars(alpha = 0.13)
reg.fit(data, labels)
print reg.coef_
preds = reg.predict(testdata)
lassolars_ls = mean_squared_error(testlabels, preds)
print 'LARS_LS =', lassolars_ls
'''
'''
reg = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
reg.fit(data, labels)
print reg.coef_
preds = reg.predict(testdata)
omp_ls = mean_squared_error(testlabels, preds)
print 'OMP_LS =', omp_ls
'''

reg = ElasticNet(alpha = 7.18)
reg.fit(data, labels)
print reg.coef_
preds = reg.predict(testdata)
en_ls = mean_squared_error(testlabels, preds)
print 'EN_LS =', en_ls
