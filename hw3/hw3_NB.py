"""
Python Code for HW3 Problem2
Naive Bayes

Created on Wed Oct 12 2016
@author: Yizhou Wang

"""

import csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from random import shuffle
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB




"""
function ( error_mean ) = cross_validation( feature_matrix, data_tr_labels )
Cross-validation for Naive Bayes.

    Input:
        feature_matrix 	feature representation matrix
        data_tr_labels	training dataset labels
        
    Output:
        error_mean    	CV test mean error rate in 5 folds

@author: Yizhou Wang

"""

def cross_validation ( feature_matrix, data_tr_labels ):

	kf = KFold(n_splits=5)
	folds = 0
	error = np.zeros(5)
	for train, test in kf.split(feature_matrix):
		
		print '---------- FOLD', folds+1, '----------'
		X_train = feature_matrix[train]
		X_test = feature_matrix[test]
		y_train = data_tr_labels[train]
		y_test = data_tr_labels[test]

		(nd_train,nf) = X_train.shape
		(nd_test,nf) = X_test.shape

		print 'Naive Bayes Training...'
		gnb = MultinomialNB()
		fit = gnb.fit(X_train, y_train)

		print 'Naive Bayes Testing...'
		preds = gnb.predict(X_test)


		# print preds.shape

		print 'Calculating Error Rate...'
		error[folds] = np.sum((y_test!=preds).astype(float))/nd_test
		print 'Error Rate =', error[folds]
		folds = folds + 1

	error_mean = np.mean(error)
	return error_mean

"""
END of cross_validation
"""








print '\n***** Learning Method: Naive Bayes *****\n'

print 'Loading Training Data...'

with open('reviews_tr.csv', 'r') as csvfile:
	data_tr_reader = csv.reader(csvfile)
	data = [data for data in data_tr_reader]

data_tr = np.asarray(data)
data_tr_labels = data_tr[1::,0]
data_tr_labels = [int(numstr) for numstr in data_tr_labels]
data_tr_text = data_tr[1::,1]


data_tr_labels = np.array(data_tr_labels)
'''
# Change labels 0s to -1s
data_tr_labels[data_tr_labels == 0] = -1
'''
ntraindata = len(data_tr_labels)

print 'Training Data Loaded!'

'''
print 'Selecting Training Data (optional) ...'

numseleted = 200000
data_tr_labels = data_tr_labels[0:numseleted]
data_tr_text = data_tr_text[0:numseleted]

# print data_tr_labels
# print data_tr_text
print 'Training Data Selected!'
'''

print '\n**************************************************\n'

print '-- Training Data Representation --'

print 'Generating Unigram Representation...'
vec_uni = CountVectorizer()
unigram = vec_uni.fit_transform(data_tr_text)
print 'Unigram Representation Generated!'

print '\n**************************************************\n'

print '-- Cross-validation --'

ave_error = np.zeros(4)
print 'Cross-validation for Unigram...'
ave_error = cross_validation ( unigram, data_tr_labels )


print '######################################################################'
print '### Cross-validation Error Rates for each Data Representation is:'
print '###', ave_error
print '######################################################################'


print '\n**************************************************\n'

print 'Training Classifier using Naive Bayes...'

print 'Unigram Chosen! Training...'
gnb_tr = MultinomialNB()
fit = gnb_tr.fit(unigram, data_tr_labels)

print 'Naive Bayes fit Finished!'

print '\n**************************************************\n'

print '-- Calculate Training Error Rate --'

print 'Testing...'

preds_tr = gnb_tr.predict(unigram)

print 'Calculating Error Rate...'

error_tr = np.sum((data_tr_labels!=preds_tr).astype(float))/numseleted

print '######################################################################'
print '### Training Error Rate is:', error_tr
print '######################################################################'


print '\n**************************************************\n'

print 'Loading Test Data...'

with open('reviews_te.csv', 'r') as csvfile:
	data_te_reader = csv.reader(csvfile)
	data = [data for data in data_te_reader]

data_te = np.asarray(data)
data_te_labels = data_te[1::,0]
data_te_labels = [int(numstr) for numstr in data_te_labels]
data_te_text = data_te[1::,1]

ntestdata = len(data_te_labels)

data_te_labels = np.array(data_te_labels)
'''
# Change labels 0s to -1s
data_te_labels[data_te_labels == 0] = -1
'''

print 'Test Data Loaded!'

print '\n**************************************************\n'

print '-- Test Data Representation --'

print 'Generating Unigram Representation...'
feature_te = vec_uni.transform(data_te_text)
print 'Unigram Representation Generated!'


print '\n**************************************************\n'

print '-- Calculate Test Error Rate --'

print 'Testing...'

preds_te = gnb_tr.predict(feature_te)

print 'Calculating Error Rate...'

error_te = np.sum((data_te_labels!=preds_te).astype(float))/ntestdata

print '######################################################################'
print '### Test Error Rate is:', error_te
print '######################################################################'

print '\n'







