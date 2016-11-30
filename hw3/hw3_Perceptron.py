"""
Python Code for HW3 Problem2
Averaged-Perceptron

Created on Wed Oct 09 2016
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




"""
function ( tfidf ) = tfidfWeight( unigram )
Calculate TF-IDF Weight of text using Unigram.

    Input:
        unigram   	Unigram Representation result (Sparse Matrix)
        
    Output:
        tfidf    	TF-IDF Weight Representation (Sparse Matrix)

@author: Yizhou Wang

"""

def tfidfWeight ( unigram ):

	(nD, nfeature_uni) = unigram.shape

	# nD_containw = np.zeros(nfeature_uni)
	# for f in range(nfeature_uni):
	# 	# print np.count_nonzero(unigram[:,f].toarray())
	# 	# print nD_containw
	# 	nD_containw[f] = np.count_nonzero(unigram[:,f].toarray())

	nD_containw = np.bincount(unigram.indices)

	idf = float(nD) / nD_containw
	lg_idf = np.log10(idf)
	# lg_idf = np.tile(lg_idf,(nD,1))
	# print unigram.shape
	# print lg_idf.shape

	# tfidf = unigram * np.diag(lg_idf)
	temp = sparse.diags(lg_idf)
	tfidf = unigram * temp

	# tfidf = sparse.csr_matrix(tfidf) 

	return tfidf

"""
END of tfidfWeight
"""




"""
function ( a ) = csr_append( a, b )
Append two sparse matrices together herizontally.

    Input:
        a, b   	Two Sparse matrices with same number of columns
        
    Output:
        a    	Append result

@author: Yizhou Wang

"""

def csr_append( a, b ):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])

    return a

"""
END of csr_append
"""


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

		print 'Average Perceptron Training...'
		(w,b) = ave_perceptron_train ( X_train, y_train )

		print 'Average Perceptron Testing...'
		preds = perceptron_test(w, b, X_test)

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





print '\n***** Learning Method: Perceptron *****\n'

print 'Loading Training Data...'

with open('reviews_tr.csv', 'r') as csvfile:
	data_tr_reader = csv.reader(csvfile)
	data = [data for data in data_tr_reader]

data_tr = np.asarray(data)
data_tr_labels = data_tr[1::,0]
data_tr_labels = [int(numstr) for numstr in data_tr_labels]
data_tr_text = data_tr[1::,1]

# Change labels 0s to -1s
data_tr_labels = np.array(data_tr_labels)
data_tr_labels[data_tr_labels == 0] = -1

ntraindata = len(data_tr_labels)

print 'Training Data Loaded!'

'''
print 'Selecting Training Data (optional) ...'

numseleted = 2000
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
# print unigram.shape


print 'Generating TFIDF Weighting...'
# vec_tfidf = TfidfVectorizer()
# tfidf = vec_tfidf.fit_transform(data_tr_text)
tfidf = tfidfWeight(unigram)
print 'TFIDF Weighting Generated!'


print 'Generating Bigram Representation...'
vec_bi = CountVectorizer(ngram_range=(2, 2))
bigram = vec_bi.fit_transform(data_tr_text)
print 'Bigram Representation Generated!'
# print bigram.shape

print 'Generating n-gram Representation...'
# n = 3
vec_n = CountVectorizer(ngram_range=(1, 2))
ngram = vec_n.fit_transform(data_tr_text)
print 'n-gram Representation Generated!'
# print ngram.shape


print '\n**************************************************\n'

print '-- Cross-validation --'

ave_error = np.zeros(4)

print '----------------------'
print 'Cross-validation for Unigram...'
ave_error[0] = cross_validation ( unigram, data_tr_labels )

print '----------------------'
print 'Cross-validation for TFIDF...'
ave_error[1] = cross_validation ( tfidf, data_tr_labels )

print '----------------------'
print 'Cross-validation for Bigram...'
ave_error[2] = cross_validation ( bigram, data_tr_labels )
print '----------------------'
print 'Cross-validation for n-gram...'
ave_error[3] = cross_validation ( ngram, data_tr_labels )



print '######################################################################'
print '### Cross-validation Error Rates for each Data Representation is:'
print '###', ave_error
print '######################################################################'


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

with open('reviews_te.csv', 'r') as csvfile:
	data_te_reader = csv.reader(csvfile)
	data = [data for data in data_te_reader]

data_te = np.asarray(data)
data_te_labels = data_te[1::,0]
data_te_labels = [int(numstr) for numstr in data_te_labels]
data_te_text = data_te[1::,1]

ntestdata = len(data_te_labels)

# Change labels 0s to -1s
data_te_labels = np.array(data_te_labels)
data_te_labels[data_te_labels == 0] = -1

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







