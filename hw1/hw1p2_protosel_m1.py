# -*- coding: utf-8 -*-
"""
Python Code for HW1 Problem2 method1

Created on Thu Sep 15 13:55:22 2016
@author: Yizhou Wang

"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pylab as pl
from scipy import log 
from scipy.optimize import curve_fit
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
function dist = selfdist( X )
Calculate self distance with Euclidean distance
    
    Input:
        X   matrix to calculate distance

    Output:
        dist   distance between each vector

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
    dist = sqedist
            
    return dist
    
"""
END of onennc
"""

"""
function idx = getidx( L, k, norf )
Get index of k max/min value in L.

    Input:
        L   vector of value
        k   number of max/min needed
        norf    nearest for max and farrest for min
        
    Output:
        idx     index of max/min value (dtype: list)
        
@author: Yizhou Wang
        
"""
def getidx(L,k,norf):
    
    idx = []
    
    if norf == 'nearest':
        inf = 1000000000000000
        for i in range(k):
            idx.append(np.argmin(L,axis=1))
            L[0,np.argmin(L,axis=1)] = inf
    
    if norf == 'farrest':
        nan = -1000000000000000
        for i in range(k):
            idx.append(np.argmax(L,axis=1))
            L[0,np.argmax(L,axis=1)] = nan

    return idx

"""
END of getidx
"""


"""
function nfp = knfn( X, Y, k )
k-nearest/farrest neighbor with Euclidean distance
    
    This function implements the k-nearest/farrest neighbor of X with 
    Euclidean distance. Used in Prototype Selection!!!

    Input:
        X   parent point for PS 
        Y   points to be picked
        k   number of nearest/farrest neighbor
        norf    nearest for max and farrest for min

    Output:
        nfp     nearest/farrest points

@author: Yizhou Wang

"""
def knfn(X,Y,k,norf):
    
    sn = X.shape[0] 
    tn = Y.shape[0]

    YT = Y.transpose()    
    XYT = np.dot(X,YT)
    
    sqY = Y * Y
    sumsqY = np.sum(sqY, axis=1)
    sumsqYex = np.tile(sumsqY, (sn, 1))   
    
    sqedist = sumsqYex - 2*XYT 
    dist = sqedist

    k = int(k)    
    
    if k == 1:
        if norf == 'nearest':        
            nearidx = np.array(getidx(dist,1,'nearest'))
            nearp = Y[nearidx.transpose(),:][0]
            nfp = nearp
        if norf == 'farrest':
            faridx = np.array(getidx(dist,1,'farrest'))
            farp = Y[nearidx.transpose(),:][0]
            nfp = farp
    else:
        if norf == 'nearest':     
            nearidx = np.array(getidx(dist,k,'nearest'))
            nearp = Y[nearidx.transpose(),:][0]
            nfp = nearp
        if norf == 'farrest':
            faridx = np.array(getidx(dist,k,'farrest'))
            farp = Y[faridx.transpose(),:][0]
            nfp = farp
        if norf == 'n&f':
            nearidx = np.array(getidx(dist,(k+1)/2-1,'nearest'))
            nearp = Y[nearidx.transpose(),:][0]
            faridx = np.array(getidx(dist,(k+1)/2,'farrest'))
            farp = Y[faridx.transpose(),:][0]
            nfp = np.concatenate((nearp,farp))
    
    return nfp
    
"""
END of knfn
"""



"""
function ( sp_list, sp_mean ) = splitmean( X, Y )
Split data according to labels, and calculate the mean point of each label.

    Input:
        X   training data set
        Y   training data labels
        
    Output:
        sp_list    Split data list
        sp_mean    Split data mean points

@author: Yizhou Wang

"""
def splitmean(X,Y):

    sn,ss = X.shape

    sp_list = []
    sp_mean = np.zeros((10,ss))
    
    for i in range(0,10):
        idx = [idx for idx, e in enumerate(Y) if e==i]
        sp_list.append(X[idx,:])
        sp_mean[i,:] = np.mean(X[idx,:],axis=0)
    
    return (sp_list, sp_mean)

"""
END of splitmean
"""



"""
function ( protoX, protoY ) = protosel( X, Y, m )
Prototype selection on a training data set.

    Prototype selection is a method for speeding-up nearest neighbor search 
    that replaces the training data with a smaller subset of prototypes.
    
    Input:
        X    training data set
        Y    training data labels
        m    select number
        
    Output:
        protoX  smaller training data set after PS
        protoY  smaller training data labels after PS

@author: Yizhou Wang

"""
def protosel(X,Y,m):
    
    (sp_list_raw,sp_mean) = splitmean(X,Y)
    sp_list = []    
    
    for i in range(0,10):    
        distY = selfdist(sp_list_raw[i])
        eyen = sp_list_raw[i].shape[0]
        distY_min_idx = np.argmin(distY+10000000000000*np.eye(eyen),axis=0)
        sp_list.append(np.delete(sp_list_raw[i],distY_min_idx,axis=0))
        
    protoX = sp_mean 
    protoY = (np.array([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]])).transpose()
    
    fpm = np.zeros((10,ss))
    
    ratio = 0.5
    
    for i in range(0,10):
        
        nfp = knfn(np.array([sp_mean[1,:]]),sp_list[i],ratio*m/10-1,'n&f')
        
        for listidx in range(0,10):        
            fpm[listidx,:] = knfn(sp_mean,sp_list[i],1,'nearest')    

        cencirp = knfn(fpm,sp_list[i],(1-ratio)*m/10,'nearest')
        
        protoX = np.concatenate((protoX, nfp))
        protoX = np.concatenate((protoX, cencirp))
        protoY = np.concatenate((protoY, np.tile([i],(m/10-1,1))))
    
    return (protoX,protoY)

"""
END of protosel
"""





n = [1000, 2000, 4000, 8000]
error = np.zeros((4,10))

for i in range(0,4):
    for time in range(0,10):
        (protoX,protoY) = protosel(data,labels,n[i]/5*2)        

        sel = random.sample(xrange(60000),n[i]/5*3)
        protoX = np.concatenate((protoX, data[sel,:])) 
        protoY = np.concatenate((protoY, labels[sel]))
        
        preds = onennc(protoX,protoY,testdata)[0]
#        error[i,0] = np.sum((testlabels!=preds).astype(float))/10000.0
#        print error[i,0]

        error[i,time] = np.sum((testlabels!=preds).astype(float))/10000.0
        print error[i,time]

error_ave = np.mean(error,axis=1)
print error_ave
error_std = np.std(error,axis=1)
print error_std








"""
Draw a plot of test points and learning curve
The Learning Curve is generated by logistic fit
"""

"""
function para = polyfit( x, y )
Logistic fit using the input points.

    Input:
        x   the x-axis of input points
        y   the y-axis of input points
        
    Output:
        para    the parameters of LOG function a, b 
                a, b is discribed in function func
                
@author: Yizhou Wang
                
"""

def func(x, a, b):
    y = a * log(x) + b
    return y

def polyfit(x, y):
    popt, pcov = curve_fit(func, x, y)
    para = popt
    return para

"""
END of polyfit
"""

plt.errorbar(n, error_ave, yerr=error_std, fmt='o',label='test points')

lx = np.arange(0,60000,100)
para = polyfit(n,error_ave)
ly = para[0] * log(lx) + para[1]
plt.plot(lx,ly,label='learning curve')

pl.title('Learning Curve Plot')
pl.xlabel('Number of Training Data n')
pl.ylabel('Test Error Rate')
pl.xlim(0.0,60000.)
pl.ylim(0.0,0.15)

plt.legend()
plt.show()




