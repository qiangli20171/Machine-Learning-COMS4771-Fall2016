#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Python Code for HW2 Problem1b

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


ny = 20
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
def split ( X, Y ):

    sn,ss = X.shape
    sp_list = []
    
    for i in range(1,21):
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

	sp_list = split(data,labels)

	for y in range(ny):
		sp_array = sp_list[y].toarray()
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



"""
function ( error ) = gm_test( data, labels, mu, pi, ny )
Generative Models training function based on naive bayes classifiers.
This function is to get error rate of previous trained classifier. 

    Input:
        data   	training data set
        labels  training data labels
        mu		class conditional distribution parameters
        pr 		class prior parameters
        ny 		number of catagories
        
    Output:
        error	test error rate

@author: Yizhou Wang

"""
def gm_test ( data, labels, mu, pi, ny ):

	(ni_test,nd) = data.shape
	pro = np.zeros((ni_test,ny))

	data_ = np.tile([1],(ni_test,nd)) - data

	mu_log = np.log(mu)
	mu_log_ = np.log(1 - mu)
	
	pro = mu_log * np.transpose(data) + mu_log_ * np.transpose(data_)
	pro_w = np.zeros((ny,ni_test))
	for y in range(ny):
		pro_w[y,:] = np.log(pi[y,0]) + pro[y,:]

	# print "SHAPE(pro) = " , pro.shape
	# print "SHAPE(pro_w) = " , pro_w.shape

	preds = np.zeros((ni_test,1))
	preds[:,0] = np.argmax(pro_w,axis=0)
	# print "SHAPE(preds) = " , preds.shape
	# print preds

	error = np.sum((labels!=preds+1).astype(float))/ni_test
	# print 'Error = '
	# print error

	return error

"""
END of gm_test
"""



print 'Training Model...'
(mu, pi) = gm_train(data,labels,ny)
print 'mu = '
print mu
print 'pi = '
print pi
print 'Training Finished!'

print 'Testing Training Error Rate...'
er_train = gm_test(data,labels,mu,pi,ny)
print 'Test Finished!'
print 'Training Error Rate = '
print er_train

print 'Testing Test Error Rate...'
er_test = gm_test(testdata,testlabels,mu,pi,ny)
print 'Test Finished!'
print 'Test Error Rate = '
print er_test


