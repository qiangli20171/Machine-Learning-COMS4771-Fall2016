# hw5p1_nn.py

"""
Python Code for HW5 Problem1
Neural Networks

Created on Wed Nov 22 2016
@author: Yizhou Wang

"""

import csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from random import shuffle
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

from hw5p1_utils import load_data
from hw5p1_utils import csr_append




'''
"""
function ( u, beta ) = ave_perceptron_train( data, labels )
Averaged-Perceptron Algorithm with the average of the latter half data.

    Input:
        data   	Dataset after feature representation (sparse matrix)
        labels 	Data labels with 1 and -1
        
    Output:
        u    	Averaged weight
        beta	Averaged bias

	Some important variables:
		nd 		the number of data
		nf 		the number of feature
		X 		data after shuffle
		Y 		lables after shuffle
		w 		linear classifiers
		b 		thresholds (bias)

@author: Yizhou Wang

"""

def ave_perceptron_train ( data, labels ):
	
	(nd, nf) = data.shape

	arr = np.arange(nd)
	shuffle(arr)
	X1 = data[arr,:]
	Y1 = labels[arr]
	arr = np.arange(nd)
	shuffle(arr)
	X2 = data[arr,:]
	Y2 = labels[arr]
	X = csr_append(X1, X2)
	Y = np.append(Y1, Y2, axis=0)

	w = np.zeros((1,nf))
	b = 0
	u = np.zeros((1,nf))
	beta = 0

	# cw = 1
	ww = 1
	for t in tqdm(range(2*nd)):

		judge = Y[t] * (np.dot(w, X[t,:].toarray().transpose()) + b)

		if judge <= 0:
			w = w + Y[t] * X[t,:]
			b = b + Y[t]
			
			if t >= nd:
				u = u + w * ww
				beta = beta + b * ww
				ww = 1
				# cw = cw + 1
		else:
			if t >= nd:
				ww = ww + 1

	return u, beta

"""
END of ave_perceptron_train
"""


"""
function ( preds ) = perceptron_test( w, b, X )
Perceptron test function.

    Input:
        w, b   	Perceptron training result
        X 		Test dataset
        
    Output:
        preds   Predict labels

@author: Yizhou Wang

"""

def perceptron_test ( w, b, X ):

	(nd_test,nf) = X.shape
	preds = np.zeros(nd_test)

	for i in tqdm(range(nd_test)):
		preds[i] = np.dot(w, X[i,:].toarray().transpose()) + b

	return np.sign(preds)

"""
END of perceptron_test
"""
'''


"""
function ( error_mean ) = cross_validation( feature_matrix, data_tr_labels )
Cross-validation for Averaged-perceptron.

    Input:
        feature_matrix 	feature representation matrix
        data_tr_labels	training dataset labels
        
    Output:
        error_mean    	CV test mean error rate in 5 folds

@author: Yizhou Wang

"""

def valid_nn ( feature_matrix, data_tr_labels, hidden_layer_sizes, valid_model='holdout' ):

	if valid_model == 'holdout':
		X_train, X_test, y_train, y_test = train_test_split(
			feature_matrix, 
			data_tr_labels, 
			test_size=0.2, 
			random_state=0
		)

		(nd_train,nf) = X_train.shape
		(nd_test,nf) = X_test.shape

		clf = MLPClassifier(
			hidden_layer_sizes=hidden_layer_sizes,
			activation='relu', 
			learning_rate_init=0.001, 
			max_iter=100, 
			tol=0.0001, 
			verbose=True, 
			momentum=0.9
		)

		print '... Neural Network Training ...'
		clf.fit(X_train, y_train)
		print '... Neural Network Predicting ...'			
		preds = clf.predict(X_test)

		print '... Calculating Error Rate...'
		error = np.sum((y_test!=preds).astype(float))/nd_test
		print 'Error Rate =', error

		return error


	if valid_model == 'kfold':
		kf = KFold(n_splits=5)
		folds = 0
		error = np.zeros(5)
		for train, test in kf.split(feature_matrix):
			
			print '\n---------- FOLD', folds+1, '----------'
			X_train = feature_matrix[train]
			X_test = feature_matrix[test]
			y_train = data_tr_labels[train]
			y_test = data_tr_labels[test]

			(nd_train,nf) = X_train.shape
			(nd_test,nf) = X_test.shape

			clf = MLPClassifier(
				hidden_layer_sizes=hidden_layer_sizes,
				activation='relu', 
				learning_rate_init=0.001, 
				max_iter=200, 
				tol=0.0001, 
				verbose=True, 
				momentum=0.9
			)

			print '... Neural Network Training ...'
			clf.fit(X_train, y_train)
			print '... Neural Network Predicting ...'
			preds = clf.predict(X_test)

			print '... Calculating Error Rate...'
			error[folds] = np.sum((y_test!=preds).astype(float))/nd_test
			print 'Error Rate =', error[folds]
			folds = folds + 1

		error_mean = np.mean(error)
		return error_mean

"""
END of cross_validation
"""

def valid_rf ( feature_matrix, data_tr_labels, n_trees ):

	X_train, X_test, y_train, y_test = train_test_split(
		feature_matrix, 
		data_tr_labels, 
		test_size=0.2, 
		random_state=0
	)

	(nd_train,nf) = X_train.shape
	(nd_test,nf) = X_test.shape

	clf = RandomForestClassifier( n_estimators=n_trees, verbose=n_trees )

	print '... Random Forest Training ...'
	clf.fit(X_train, y_train)
	print '... Random Forest Predicting ...'			
	preds = clf.predict(X_test)

	print '... Calculating Error Rate...'
	error = np.sum((y_test!=preds).astype(float))/nd_test
	print 'Error Rate =', error

	return error




print '... Loading Training Data ...'

data_tr_text, data_tr_labels = load_data ( 
	filename='reviews_tr.csv', 
	SelectOrNot=False, 
	numseleted=200000
)
'''
# Change labels 0s to -1s
data_tr_labels[data_tr_labels == 0] = -1
'''
ntraindata = len(data_tr_labels)

print 'Training Data Loaded!'

print '\n... Training Data Representation ...'
'''
print 'Generating TFIDF Weighting...'
vec_tfidf = TfidfVectorizer()
fr = vec_tfidf.fit_transform(data_tr_text)
print 'TFIDF Weighting Generated!'

'''
print '... Generating Bigram Representation ...'
vec_bi = CountVectorizer(ngram_range=(1, 2))
fr = vec_bi.fit_transform(data_tr_text)
print 'Bigram Representation Generated!'


clf = RandomForestClassifier( n_estimators=4, verbose=4 )

print '... Random Forest Training ...'
clf.fit(fr, data_tr_labels)
print '... Random Forest Predicting ...'			
preds = clf.predict(fr)

print '... Calculating Error Rate...'
error = np.sum((data_tr_labels!=preds).astype(float))/ntraindata
print 'Error Rate =', error



print '\n... Cross-validation ...'

ave_error = np.zeros(12)

print '\n... Tree Size = 16 ...'
ave_error[8] = valid_rf ( fr, data_tr_labels, 16 )
print '\n... Tree Size = 32 ...'
ave_error[9] = valid_rf ( fr, data_tr_labels, 32 )
# print '\n... Tree Size = 256 ...'
# ave_error[10] = valid_rf ( fr, data_tr_labels, 256 )
# print '\n... Tree Size = 32 ...'
# ave_error[11] = valid_rf ( fr, data_tr_labels, 32 )


# print '\n... Hidden Layer Size = (50, ) ...'
# ave_error[0] = valid_nn ( fr, data_tr_labels, (5, ) )
# print '\n... Hidden Layer Size = (100, ) ...'
# ave_error[1] = valid_nn ( fr, data_tr_labels, (10, ) )
# print '\n... Hidden Layer Size = (200, ) ...'
# ave_error[2] = valid_nn ( fr, data_tr_labels, (20, ) )
# print '\n... Hidden Layer Size = (500, ) ...'
# ave_error[3] = valid_nn ( fr, data_tr_labels, (500, ) )

# print '\n... Hidden Layer Size = (50, 50) ...'
# ave_error[4] = valid_nn ( fr, data_tr_labels, (5, 5) )
# print '\n... Hidden Layer Size = (100, 100) ...'
# ave_error[5] = valid_nn ( fr, data_tr_labels, (10, 10) )
# print '\n... Hidden Layer Size = (200, 200) ...'
# ave_error[6] = valid_nn ( fr, data_tr_labels, (20, 20) )
# print '\n... Hidden Layer Size = (500, 500) ...'
# ave_error[7] = valid_nn ( fr, data_tr_labels, (500, 500) )

# print '\n... Hidden Layer Size = (50, 50, 50) ...'
# ave_error[8] = valid_nn ( fr, data_tr_labels, (5, 5, 5) )
# print '\n... Hidden Layer Size = (100, 100, 100) ...'
# ave_error[9] = valid_nn ( fr, data_tr_labels, (10, 10, 10) )
# print '\n... Hidden Layer Size = (200, 200, 200) ...'
# ave_error[10] = valid_nn ( fr, data_tr_labels, (20, 20, 20) )
# print '\n... Hidden Layer Size = (500, 500, 500) ...'
# ave_error[11] = valid_nn ( fr, data_tr_labels, (500, 500, 500) )


print '######################################################################'
print 'Cross-validation Error Rate for Data Representation is:'
print ave_error
print '######################################################################'

'''
print '\n**************************************************\n'

print 'Training Classifier using Perceptron...'

repre2choose = np.argmin(ave_error)
if repre2choose == 0:
	print 'Unigram Chosen! Training...'
	(w,b) = ave_perceptron_train ( unigram, data_tr_labels )
elif repre2choose == 1:
	print 'TFIDF Chosen! Training...'
	(w,b) = ave_perceptron_train ( tfidf, data_tr_labels )
elif repre2choose == 2:
	print 'Bigram Chosen! Training...'
	(w,b) = ave_perceptron_train ( bigram, data_tr_labels )
elif repre2choose == 3:
	print 'gram Chosen! Training...'
	(w,b) = ave_perceptron_train ( ngram, data_tr_labels )

# (w,b) = ave_perceptron_train ( tfidf, data_tr_labels )

print 'Averaged-Perceptron Finished!'

print '\n**************************************************\n'

print '-- Calculate Training Error Rate --'

print 'Testing...'

if repre2choose == 0:
	preds_tr = perceptron_test(w, b, unigram)
elif repre2choose == 1:
	preds_tr = perceptron_test(w, b, tfidf)
elif repre2choose == 2:
	preds_tr = perceptron_test(w, b, bigram)
elif repre2choose == 3:
	preds_tr = perceptron_test(w, b, ngram)

# preds_tr = perceptron_test(w, b, tfidf)

print 'Calculating Error Rate...'

error_tr = np.sum((data_tr_labels!=preds_tr).astype(float))/numseleted

print '######################################################################'
print '### Training Error Rate is:', error_tr
print '######################################################################'


print '\n**************************************************\n'

print 'Loading Test Data...'

data_te_text, data_te_labels = load_data ( 'reviews_te.csv' )

# Change labels 0s to -1s
data_te_labels = np.array(data_te_labels)
data_te_labels[data_te_labels == 0] = -1

ntestdata = len(data_te_labels)

print 'Test Data Loaded!'

print '\n**************************************************\n'

print '-- Test Data Representation --'

if repre2choose == 0:
	print 'Generating Unigram Representation...'
	feature_te = vec_uni.transform(data_te_text)
	print 'Unigram Representation Generated!'
elif repre2choose == 1:
	print 'Generating TFIDF Weighting...'
	# feature_te = vec_tfidf.transform(data_te_text)
	unigram = vec_uni.transform(data_te_text)
	tfidf = tfidfWeight(unigram)
	print 'TFIDF Weighting Generated!'
elif repre2choose == 2:
	print 'Generating Bigram Representation...'
	feature_te = vec_bi.transform(data_te_text)
	print 'Bigram Representation Generated!'
elif repre2choose == 3:
	print 'Generating n-gram Representation...'
	feature_te = vec_n.transform(data_te_text)
	print 'n-gram Representation Generated!'

# feature_te = vec_tfidf.transform(data_te_text)

print '\n**************************************************\n'

print '-- Calculate Test Error Rate --'

print 'Testing...'

preds_te = perceptron_test ( w, b, feature_te )

print 'Calculating Error Rate...'


error_te = np.sum((data_te_labels!=preds_te).astype(float))/ntestdata

print '######################################################################'
print '### Test Error Rate is:', error_te
print '######################################################################'

print '\n'





'''

