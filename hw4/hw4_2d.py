"""
Python Code for HW4 Problem2

Created on Wed Oct 28 2016
@author: Yizhou Wang

"""

import numpy as np
from scipy.io import loadmat
from math import floor



"""
function ( object_sum ) = func( X, Y, beta0, beta )
Compute function value.

    Input:
        X, Y 	data set and labels
        beta0	bias
        beta 	transform
        
    Output:
        object_sum    	object function value

@author: Yizhou Wang

"""

def func ( X, Y, beta0, beta ):

	(n, d) = X.shape
	betaX = np.dot(X, beta)
	beta0_betaX = beta0 + betaX

	ln_part = np.log(1 + np.exp(beta0_betaX))
	Y = np.squeeze(np.asarray(Y))
	yi_part = Y * beta0_betaX

	object_value = ln_part - yi_part
	object_sum = np.sum(object_value) / n

	return object_sum



"""
function ( grad_beta0, grad_beta ) = gradfunc( X, Y, beta0, beta )
Compute gradient of function.

    Input:
        X, Y 	data set and labels
        beta0	bias
        beta 	transform
        
    Output:
        grad_beta0    	gradient with respect to beta0
        grad_beta 		gradient with respect to beta

@author: Yizhou Wang

"""

def gradfunc ( X, Y, beta0, beta ):

	(n, d) = X.shape
	tmp = np.exp(beta0) * np.exp(np.dot(X, beta))
	Y = np.squeeze(np.asarray(Y))

	tmp1 = np.zeros(n)
	tmp2 = np.zeros((n, d))
	for i in range(n):
		tmp1[i] = tmp[i] / (1 + tmp[i]) - Y[i]
		tmp2[i, :] = (tmp[i] / (1 + tmp[i]) - Y[i]) * X[i, :]

	grad_beta0 = np.sum(tmp1) / n
	grad_beta = np.sum(tmp2, axis=0) / n

	return (grad_beta0, grad_beta)




"""
function ( preds ) = lc_test( beta0, beta, X )
Test function for gd.

    Input:
        beta0		bias
        beta		transform
        X 			test data set
        
    Output:
        preds 		test results

@author: Yizhou Wang

"""

def lc_test ( beta0, beta, X ):

	(nd_test,nf) = X.shape
	preds = np.zeros(nd_test)

	for i in range(nd_test):
		tmp = np.dot(beta, X[i,:]) + beta0
		if tmp >= 0:
			preds[i] = 1
		else:
			preds[i] = 0

	return preds



"""
function ( beta0_new, beta_new ) = grad_descent( X, Y, init_beta0, init_beta, step_size, n_iteration )
Gradient Descent Algorithm.

    Input:
        X, Y 			data set and labels
        init_beta0		initial value of bias
        init_beta 		initial value of transform
        step_size 		step size
        n_iteration		number of iterations
        
    Output:
        beta0_new    	gd result of beta0
        beta_new 		gd result of beta

@author: Yizhou Wang

"""

def grad_descent ( X, Y, init_beta0, init_beta, step_size, n_iteration ):

	'''Split Dataset'''
	(n, d) = X.shape
	
	n_train = int(floor(0.8 * n))
	n_test = n - n_train

	X_train = X[0:n_train, :]
	X_test = X[n_train:, :]

	Y_train = Y[0:n_train, :]
	Y_test = np.squeeze(np.asarray(Y[n_train:, :]))
	
	'''Gradient Descent'''
	beta0_new = init_beta0
	beta_new = init_beta
	lambda0 = step_size

	t = 5  # 2^5 = 32
	error_old = 1.

	for iter_time in range(n_iteration):

		beta0 = beta0_new
		beta = beta_new
		grad_f = gradfunc ( X_train, Y_train, beta0, beta )

		'''Compute Step Size'''
		step_size = lambda0
		f_tmp1 = func ( X_train, Y_train, beta0 - step_size * grad_f[0], beta - step_size * grad_f[1] )
		f_tmp2 = func ( X_train, Y_train, beta0, beta ) - 0.5 * step_size * (grad_f[0] ** 2 + np.dot(grad_f[1],grad_f[1]))

		while f_tmp1 > f_tmp2:
			step_size = step_size / 2.0
			f_tmp1 = func ( X_train, Y_train, beta0 - step_size * grad_f[0], beta - step_size * grad_f[1] )
			f_tmp2 = func ( X_train, Y_train, beta0, beta ) - 0.5 * step_size * (grad_f[0] ** 2 + np.dot(grad_f[1],grad_f[1]))

		beta0_new = beta0 - step_size * grad_f[0]
		beta_new = beta - step_size * grad_f[1]

		f_value = func ( X_train, Y_train, beta0_new, beta_new )
		# print '... epoch', iter_time + 1, ':', 'f =', f_value, '; grad_f =', grad_f, '; StepSize =', step_size
		# print beta0_new, beta_new

		if iter_time + 1 == 2 ** t:
			t = t + 1
			preds = lc_test ( beta0_new, beta_new, X_test )
			error = np.sum((Y_test!=preds).astype(float)) / n_test

			print '... epoch', iter_time + 1, ':', 'ErrorRate =', error, '; f =', f_value, '; grad_f =', grad_f

			# print '!!!!!!!!!!!!'
			# print np.sum((Y_test!=preds).astype(float))
			# print Y_test.shape
			# print preds.shape
			# # print preds
			# print error
			# print '!!!!!!!!!!!!'

			if error > 0.99 * error_old:
				break
			else:
				error_old = error

	return (beta0_new, beta_new, error)





hw4data = loadmat('hw4data.mat')
data = hw4data['data'].astype(float)
labels = hw4data['labels'].astype(float)
(n, d) = data.shape


'''Problem 2 (d)'''

print 'Gradient Descent Using Original Data ...\n'

opt_result_2d1 = grad_descent ( data, labels, 0, np.zeros(d), 1, 10000 )
print '\n---------------------------------------------------------------------------------'
print '- OPT Results:'
print '- beta =', opt_result_2d1[0]
print '- beta0 =', list(opt_result_2d1[1])
print '- Error Rate =', opt_result_2d1[2]
print '---------------------------------------------------------------------------------\n'


print 'Gradient Descent Using Transformed Data ...\n'

A = np.array([[1/20., 0., 0.], [0., 1., 0.], [0., 0., 1/20.]])
data_new = np.dot(A, data.T).T
# print data_new.shape

opt_result_2d2 = grad_descent ( data_new, labels, 0, np.zeros(d), 1, 10000 )
print '\n---------------------------------------------------------------------------------'
print '- OPT Results:'
print '- beta =', opt_result_2d2[0]
print '- beta0 =', list(opt_result_2d2[1])
print '- Error Rate =', opt_result_2d2[2]
print '---------------------------------------------------------------------------------\n'


