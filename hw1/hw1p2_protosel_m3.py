# -*- coding: utf-8 -*-
"""
Python Code for HW1 Problem2 method3

Created on Thu Sep 15 13:55:22 2016
@author: Yizhou Wang

"""

import numpy as np
import random
from lshash import LSHash
from scipy.io import loadmat

ocr = loadmat('ocr.mat')

data = ocr['data'].astype(float)
labels = ocr['labels'].astype(float)
testdata = ocr['testdata'].astype(float)
testlabels = ocr['testlabels'].astype(float)

sn,ss = data.shape
tn,ss = testdata.shape



"""
function preds = onennc( X, Y, test )
1-nearest neighbor classifier with Euclidean distance
    
    This function implements the 1-nearest neighbor classifier of Y with
    Euclidean distance. 

    Input:
        X   matrix of training feature vectors
        Y   vector of the corresponding labels of X
        test    matrix of test featrue vectors

    Output:
        preds   vector of predicted labels

@author: Yizhou Wang

"""
def onennc(X,Y,test):
    
    sn,ss = X.shape
    tn,ss = test.shape
    
    testT = test.transpose()    
    XtestT = np.dot(X,testT)
    
    sqX =  X * X
    sumsqX = np.matrix(np.sum(sqX, axis=1))
    sumsqXex = np.tile(sumsqX.transpose(), (1, tn))   

    sqedist = sumsqXex - 2*XtestT
    dist = sqedist
            
    indexmin = np.argmin(dist,axis=0)
    
    return Y[indexmin]
    
"""
END of onennc
"""


"""
function preds = onennc( X, Y, test )
1-nearest neighbor classifier with Euclidean distance
    
    This function implements the 1-nearest neighbor classifier of Y with
    Euclidean distance. 

    Input:
        X   matrix of training feature vectors
        Y   vector of the corresponding labels of X
        test    matrix of test featrue vectors

    Output:
        preds   vector of predicted labels

@author: Yizhou Wang

"""
def selfdist(X):
    
    test = X
    
    sn,ss = X.shape
    tn,ss = test.shape
    
    testT = test.transpose()    
    XtestT = np.dot(X,testT)
    
    sqX =  X * X
    sumsqX = np.matrix(np.sum(sqX, axis=1))
    sumsqXex = np.tile(sumsqX.transpose(), (1, tn))   
    
    sqtest = test * test
    sumsqtest = np.sum(sqtest, axis=1)
    sumsqtestex = np.tile(sumsqtest, (sn, 1))   
    
    sqedist = sumsqXex + sumsqtestex - 2*XtestT 
#    dist = np.sqrt(np.array(sqedist))
    dist = sqedist
            
    return dist
    
"""
END of onennc
"""



"""
function preds = ps_lsh( data, labels, hash_size, k, testdata)
Prototype Selection using Locality-sensitive Hashing.
    
    Input:
        data    matrix of training feature vectors
        labels  vector of the corresponding labels of data
        hash_size   size of binary bits
        k       number of prototype required
        testdata    data waiting to test

    Output:
        preds   predicted label of testdata

@author: Yizhou Wang

"""
def ps_lsh( data, labels, hash_size, k, testdata):

    sn,ss = data.shape
    
    lsh = LSHash(hash_size,ss)
    
    for train in range(sn):
        lsh.index(data[train,:],int(labels[train,0]))
    
    tablelist = []
    for i, table in enumerate(lsh.hash_tables):
        binary_hash = lsh._hash(lsh.uniform_planes[i], testdata)
        tablelist.append(table.get_list(binary_hash))
    
    tablelist = list(tablelist[0])
    tablelen = len(tablelist)
    
    candp = []
    candplabel = []
    
    for i in range(tablelen):
        if len(tablelist[i]) == 2:    
            candp.append(tablelist[i][0])
            candplabel.append(tablelist[i][1])
        else:
            candp.append(tablelist[i])
            candplabel.append(0)
    
    protoX = np.zeros((tablelen,ss))
    protoY = np.zeros((tablelen,1))
    for i in range(tablelen):
        protoX[i,:] = np.asarray(candp[i])
    protoY[:,0] = np.array(candplabel)
    
    if tablelen >= k:
        protoX = protoX[:k]
        protoY = protoY[:k]
    else:
        sel = random.sample(xrange(60000),(k-tablelen))
        protoX = np.concatenate((protoX, data[sel,:])) 
        protoY = np.concatenate((protoY, labels[sel]))
    
    preds = onennc(protoX,protoY,np.matrix(testdata))
    return float(preds)

"""
END of ps_lsh
"""






n = [1000, 2000, 4000, 8000]
error = np.zeros((4,10))

for i in range(0,4):

    preds = np.zeros((tn,1))    
    
    for t in range(tn):        
        preds[t,0] = ps_lsh( data, labels, 4, n[i], testdata[t])
        
    error[i,0] = np.sum((testlabels!=preds).astype(float))/10000.0
    print error[i,0]






