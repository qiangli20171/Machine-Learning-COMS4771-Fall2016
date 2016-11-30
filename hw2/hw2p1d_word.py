#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Python Code for HW2 Problem1d

Created on Wed Sep 28 20:01:50 2016
@author: Yizhou Wang

"""

import numpy as np
from scipy.io import loadmat

news = loadmat('news.mat')

data = news['data'].astype(float)
labels = news['labels'].astype(float)
testdata = news['testdata'].astype(float)
testlabels = news['testlabels'].astype(float)


(ni_train,nd) = data.shape
(ni_test,nd) = testdata.shape


"""
function ( sp_list ) = split( X, Y )
Split data according to labels.

    Input:
        X   training data set
        Y   training data labels
        
    Output:
        sp_list    	Split data list
		-- Returned list is composed by 20 matrix of each labeled dataset.

@author: Yizhou Wang

"""
def split ( X, Y, cata0, cata1 ):

    sn,ss = X.shape
    sp_list = []
    
    for i in range(cata0,cata1):
        idx = [idx for idx, e in enumerate(Y) if e==i]
        sp_list.append(X[idx,:])
    
    return sp_list

"""
END of splitmean
"""



"""
function ( mu, pi ) = gm_train( data, labels, ny )
Generative Models training function based on naive bayes classifiers.
This function is to get parameters mu and pi in naive bayes classifiers.

    Input:
        data   	training data set
        labels  training data labels
        ny 		number of catagories
        
    Output:
        mu		class conditional distribution parameters
        pr 		class prior parameters

@author: Yizhou Wang

"""
def gm_train ( data, labels, ny ):
	
	(ni_train,nd) = data.shape

	mu = np.zeros((ny,nd))
	pi = np.zeros((ny,1))

	sp_list = split(data,labels,0,2)

	for y in range(ny):
		sp_array = sp_list[y]
		xsum = np.sum(sp_array,axis=0)
		nn = sp_array.shape[0]
		mu[y,:] = np.divide((1 + xsum),(2 + nn))
		pi[y,0] = np.divide(nn,float(ni_train))
	
	# print 'SHAPE(mu) = ' , mu.shape
	# print 'SHAPE(pi) = ' , pi.shape

	return (mu, pi)

"""
END of gm_train
"""


def create_data ( data, labels ):

	sp_list = split(data,labels,1,21)

	class0 = np.append(sp_list[0].toarray(),sp_list[15].toarray(), axis=0)
	class0 = np.append(class0,sp_list[19].toarray(), axis=0)
	class1 = np.append(sp_list[16].toarray(),sp_list[17].toarray(), axis=0)
	class1 = np.append(class1,sp_list[18].toarray(), axis=0)

	ni_class0 = class0.shape[0]
	ni_class1 = class1.shape[0]

	data_bi = np.append(class0,class1, axis=0)
	labels_bi = np.append(np.zeros((ni_class0,1)),np.ones((ni_class1,1)), axis=0)

	return (data_bi, labels_bi)

"""
END of create_data
"""




print 'Creating Training Dataset...'
(data_bi,labels_bi) = create_data(data,labels)
print 'Training Dataset Created!'
print 'Creating Test Dataset...'
(testdata_bi,testlabels_bi) = create_data(testdata,testlabels)
print 'Test Dataset Created!'

print 'Training Model...'
(mu, pi) = gm_train(data_bi,labels_bi,2)
print 'mu = '
print mu
print 'pi = '
print pi
print 'Training Finished!'


fs = mu[1,:] * (1 - mu[0,:])
fm = mu[0,:] * (1 - mu[1,:])
alpha = np.log(fs / fm)

# print fs.shape
# print fm.shape
# print alpha.shape
with open('news.vocab') as w:
	words = np.array(w.readlines())

alpha_idx = sorted(range(nd), key=lambda k: alpha[k])
nega_idx =  np.array(alpha_idx[0:20])
posi_idx = np.array(alpha_idx[-20:])

print 'Positive Words are'
print words[posi_idx]
print 'Negative Words are '
print words[nega_idx]

# print words_nega
# print words_posi







