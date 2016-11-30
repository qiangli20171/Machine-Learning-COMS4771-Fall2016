"""
Python Code for HW4 Problem2

Created on Wed Oct 28 2016
@author: Yizhou Wang

"""

import numpy as np
from scipy.io import loadmat


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

	# print '??????????'
	# print np.exp(beta0)
	# print tmp2[0,:]
	# print tmp2.shape
	# print grad_beta.shape
	# print '??????????'

	return (grad_beta0, grad_beta)



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

	beta0_new = init_beta0
	beta_new = init_beta
	eta0 = step_size

	for iter_time in range(n_iteration):

		beta0 = beta0_new
		beta = beta_new
		grad_f = gradfunc( X, Y, beta0, beta )

		'''Compute Step Size'''
		step_size = eta0
		f_tmp1 = func ( X, Y, beta0 - step_size * grad_f[0], beta - step_size * grad_f[1] )
		f_tmp2 = func ( X, Y, beta0, beta ) - 0.5 * step_size * (grad_f[0] ** 2 + np.dot(grad_f[1],grad_f[1]))

		while f_tmp1 > f_tmp2:
			step_size = step_size / 2.0
			f_tmp1 = func ( X, Y, beta0 - step_size * grad_f[0], beta - step_size * grad_f[1] )
			f_tmp2 = func ( X, Y, beta0, beta ) - 0.5 * step_size * (grad_f[0] ** 2 + np.dot(grad_f[1],grad_f[1]))

		beta0_new = beta0 - step_size * grad_f[0]
		beta_new = beta - step_size * grad_f[1]
		# print '??????????'
		# print beta
		# print grad_f[1]
		# print '??????????'

		f_value = func ( X, Y, beta0_new, beta_new )
		print '... epoch', iter_time + 1, ':', 'f =', f_value, '; grad_f =', grad_f, '; StepSize =', step_size

		if f_value < 0.65064:
			break

	return (beta0_new, beta_new)





hw4data = loadmat('hw4data.mat')
data = hw4data['data'].astype(float)
labels = hw4data['labels'].astype(float)
(n, d) = data.shape

# ans = func( data, labels, 0, np.zeros(d) )
# print ans
# ans = gradfunc( data, labels, 0, np.zeros(d) )
# print ans

'''Problem 2 (b)'''

opt_result_2b = grad_descent ( data, labels, 0, np.zeros(d), 1, 10000 )
print '\n---------------------------------------------------------------------------------'
print '- OPT Results:'
print '- beta =', opt_result_2b[0]
print '- beta0 =', list(opt_result_2b[1])
print '---------------------------------------------------------------------------------\n'




