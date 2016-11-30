"""
Python Code for HW5 Problem1
Neural Networks
Created on Wed Nov 22 2016
@author: Yizhou Wang
"""

import csv
import numpy
import numpy as np



def load_data ( filename, SelectOrNot=True, numseleted=2000 ):

    with open(filename, 'r') as csvfile:
        data_reader = csv.reader(csvfile)
        data = [data for data in data_reader]

    data = np.asarray(data)
    data_labels = data[1::,0]
    data_labels = [int(numstr) for numstr in data_labels]
    data_text = data[1::,1]
    data_labels = np.array(data_labels)

    if SelectOrNot:
        print '... Selecting Training Data (optional) ...'
        data_labels = data_labels[0:numseleted]
        data_text = data_text[0:numseleted]
        # print data_tr_labels
        # print data_tr_text
        print 'Training Data Selected!'

    return data_text, data_labels



def csr_append( a, b ):
    """ 
    function ( a ) = csr_append( a, b )
	Append two sparse matrices together herizontally.

    	Input:
        	a, b   	Two Sparse matrices with same number of columns
    	Output:
        	a    	Append result

	@author: Yizhou Wang

    Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. 
    The data, indices, and indptr still get copied.

    """

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])

    return a


