
"""
Code for Machine Learning HW0 Problem6

Compute the average squared Euclidean norm of the rows of data.

Created on Sun Sep 11 12:10:36 2016
@author: Yizhou Wang

"""

# load mat
from scipy .io import loadmat
ocr = loadmat ('ocr.mat')
# transform datatype
ocrdata = ocr['data'].astype('float')

import numpy
# Euclidean norm of the rows of data
normvec = numpy.apply_along_axis(numpy.linalg.norm,1,ocrdata)
# squared
squarednormvec = normvec*normvec
# average squared Euclidean norm
normmean = numpy.mean(squarednormvec)

print(normmean)

