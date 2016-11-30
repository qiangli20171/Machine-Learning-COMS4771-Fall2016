import numpy as np
from scipy.io import loadmat

from sklearn.linear_model import LogisticRegression

hw5data = loadmat('hw5data.mat')
data = hw5data['data'].astype(float)
labels = hw5data['labels'].astype(float)
# labels = ravel(labels)
(ntr, d) = data.shape
testdata = hw5data['testdata'].astype(float)
testlabels = hw5data['testlabels'].astype(float)
(nte, d) = testdata.shape

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
fit = model = model.fit(data, labels.ravel())
vbeta = model.coef_
beta = model.intercept_

# check the accuracy on the training set
train_acc = model.score(data, labels)
test_acc = model.score(testdata, testlabels)

print(vbeta, beta)
# print '!!!!!', vbeta.shape
print 'training error rate =', train_acc
print 'test error rate =', test_acc

p_con = 1. / (1 + np.exp(-1 * beta - np.dot(vbeta, testdata[0:1024].T)))
print p_con
# print p_con.ravel().shape

testlabels_reshape = np.reshape(testlabels, (128, 1024))
testlabels_reshape_mean = np.mean(testlabels_reshape, axis=0)
print 'testlabels_reshape_mean.shape =', testlabels_reshape_mean.shape
# p_i = np.tile(testlabels_reshape_mean, (1, 128)).ravel()
p_i = testlabels_reshape_mean.ravel()
print p_i

mae = np.mean(abs(p_con - p_i))

print 'MAE =', mae

