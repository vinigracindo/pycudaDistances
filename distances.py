#-*- coding:utf-8 -*-

"""Utilities to evaluate pairwise distances or metrics between 2
sets of points.

"""

# Authors: Vinnicyus Gracindo <vini.gracindo@gmail.com>
# License: GNU GPL.

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from math import sqrt
import numpy
from utils import safe_asarray

#Only main Device
MAX_THREADS_PER_BLOCK = \
    drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

BLOCK_SIZE = int(sqrt(MAX_THREADS_PER_BLOCK))

# Utility Functions
def check_pairwise_arrays(X, Y, dtype=numpy.float32):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the second dimension of the two arrays is equal.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_a, n_features]

    Y : {array-like}, shape = [n_samples_b, n_features]

    Returns
    -------
    safe_X : {array-like}, shape = [n_samples_a, n_features]
        An array equal to X, guarenteed to be a numpy array.

    safe_Y : {array-like}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guarenteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    if Y is X or Y is None:
        X = Y = safe_asarray(X, dtype=dtype)
    else:
        X = safe_asarray(X, dtype=dtype)
        Y = safe_asarray(Y, dtype=dtype)
    
    if len(X.shape) < 2:
        raise ValueError("X is required to be at least two dimensional.")
    if len(Y.shape) < 2:
        raise ValueError("Y is required to be at least two dimensional.")
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))
    
    return X, Y

def euclidean_distances(X, Y=None, inverse=True):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_1, n_features]

    Y : {array-like}, shape = [n_samples_2, n_features]

    inverse: boolean, optional
        This routine will return the inverse Euclidean distances instead.

    Returns
    -------
    distances : {array}, shape = [n_samples_1, n_samples_2]

    Examples
    --------
    >>> from pycudadistances.distances import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X, inverse=False)
    array([[ 0.,  1.],
           [ 1.,  0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]], inverse=False)
    array([[ 1.        ],
           [ 1.41421356]])
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    solution_rows = X.shape[0]
    solution_cols = Y.shape[0]
    
    dx, mx = divmod(solution_cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((solution_rows, solution_cols))
    solution = solution.astype(numpy.float32)

    kernel_code_template = """
        #include <math.h>
        
        __global__ void euclidean(float *x, float *y, float *solution) {

            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NROWS)s ) ) {
            
                float result = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    float x_e = x[%(NDIM)s * idx + iter];
                    float y_e = y[%(NDIM)s * idy + iter];
                    result += pow((x_e - y_e), 2);
                }
                
                int pos = idy + %(NROWS)s * idx;
                 
                solution[pos] = sqrt(result);
                
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': solution_rows,
        'NROWS': solution_cols,
        'NDIM': X.shape[1]
    }

    mod = SourceModule(kernel_code)
    func = mod.get_function("euclidean")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return numpy.divide(1.0, (1.0 + solution)) if inverse else solution

def pearson_correlation(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_1, n_features]

    Y : {array-like}, shape = [n_samples_2, n_features]

    Returns
    -------
    distances : {array}, shape = [n_samples_1, n_samples_2]

    Examples
    --------
    >>> from pycudadistances.distances import pearson_correlation
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> pearson_correlation(X, X)
    array([[ 1., 1.],
           [ 1., 1.]])
    >>> pearson_correlation(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.39605904],
               [ 0.39605904]])
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void pearson(float *x, float *y, float *solution) {
        
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                float sum_xy, sum_x, sum_y, sum_square_x, sum_square_y;
                
                sum_x = sum_y = sum_xy = sum_square_x = sum_square_y = 0.0f;
                
                for(int iter = 0; iter < %(NDIM)s; iter ++) {
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    sum_x += x_e;
                    sum_y += y_e;
                    sum_xy += x_e * y_e;
                    sum_square_x += pow(x_e, 2);
                    sum_square_y += pow(y_e, 2);
                }
                int pos = idx + %(NCOLS)s * idy;
                float denom = sqrt(sum_square_x - (pow(sum_x, 2) / %(NDIM)s)) * sqrt(sum_square_y - (pow(sum_y, 2) / %(NDIM)s));
                if (denom == 0) {
                    solution[pos] = 0;
                } else {
                    float quot = sum_xy - ((sum_x * sum_y) / %(NDIM)s);
                    solution[pos] = quot / denom;
                }
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1]
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("pearson")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return solution

def manhattan_distances(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This distance implementation is the distance between two points in a grid
    based on a strictly horizontal and/or vertical path (that is, along the
    grid lines as opposed to the diagonal or "as the crow flies" distance.
    The Manhattan distance is the simple sum of the horizontal and vertical
    components, whereas the diagonal distance might be computed by applying the
    Pythagorean theorem.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_1, n_features]

    Y : {array-like}, shape = [n_samples_2, n_features]

    Returns
    -------
    distances : {array}, shape = [n_samples_1, n_samples_2]

    Examples
    --------
    >>> from pycudadistances.distances import manhattan_distances
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> manhattan_distances(X, X)
    array([[ 1.,  1.],
           [ 1.,  1.]])
    >>> manhattan_distances(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.25],
          [ 0.25]])
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void manhattan(float *x, float *y, float *solution) {

            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                float result = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    result += fabs((x_e - y_e));
                }
                int pos = idx + %(NCOLS)s * idy;
                solution[pos] = result;
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1]
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("manhattan")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return 1.0 - (solution / float(X.shape[1]))

def minkowski(X, Y=None, P=1):
    """
    This is the generalized metric distance. When P=1 it becomes city 
    block distance and when P=2, it becomes Euclidean distance.
    
    Computes the Minkowski distance between two vectors ``u`` and ``v``,
    defined as

    .. math::

       {||u-v||}_p = (\sum{|u_i - v_i|^p})^{1/p}.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_1, n_features]

    Y : {array-like}, shape = [n_samples_2, n_features]
    
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.

    Returns
    -------
    distances : {array}, shape = [n_samples_1, n_samples_2]
    
    Examples
   >>> from pycudadistances.distances import minkowski
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> minkowski(X, X, P=1)
   array([[ 0.,  0.],
         [ 0.,  0.]])
   >>> minkowski(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]], P=3)
    array([[ 1.98952866],
          [ 1.98952866]])
    --------
    
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        #include <stdio.h>
        
        __global__ void minkowski(float *x, float *y, float *solution) {

            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                float result = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    result += pow(fabs(x_e - y_e), %(ORDER)s);
                }
                int pos = idx + %(NCOLS)s * idy;
                solution[pos] = pow(result, 1/float(%(ORDER)s));
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1],
        'ORDER': P
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("minkowski")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return solution

def cosine_distances(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

     An implementation of the cosine similarity. The result is the cosine of
     the angle formed between the two preference vectors.
     Note that this similarity does not "center" its data, shifts the user's
     preference values so that each of their means is 0. For this behavior,
     use Pearson Coefficient, which actually is mathematically
     equivalent for centered data.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from pycudadistances.distances import cosine_distances
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> cosine_distances(X, X)
    array([[ 1.,  1.],
          [ 1.,  1.]])
    >>> cosine_distances(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.9606463],
           [ 0.9606463]])

    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void cosine(float *x, float *y, float *solution) {

            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                float sum_ab = 0.0;
                float mag_a = 0.0;
                float mag_b = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    sum_ab += x_e * y_e;
                    mag_a += pow(x_e, 2);
                    mag_b += pow(y_e, 2);
                }
                int pos = idx + %(NCOLS)s * idy;
                solution[pos] = sum_ab / (sqrt(mag_a) * sqrt(mag_b));
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1]
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("cosine")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return solution

def hamming(X, Y=None):
    """
    Computes the Hamming distance between two n-vectors ``u`` and
    ``v``, which is simply the proportion of disagreeing components in
    ``u`` and ``v``. If ``u`` and ``v`` are boolean vectors, the Hamming
    distance is

    .. math::

       \frac{c_{01} + c_{10}}{n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)
    
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void hamming(float *x, float *y, float *solution) {

            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                int diff = 0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    if(x_e != y_e) diff++;
                }
                int pos = idx + %(NCOLS)s * idy;
                solution[pos] = diff / float(%(NDIM)s);
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1]
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("hamming")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return solution

#def canberra(X, Y):
#    X, Y = check_pairwise_arrays(X,Y)
#     
#    rows = X.shape[0]
#    cols = Y.shape[0]
#    
#    solution = numpy.zeros((rows, cols))
#    solution = solution.astype(numpy.float32)
#    
#    kernel_code_template = """
#        #include <math.h>
#        
#        __global__ void canberra(float *x, float *y, float *solution) {
#
#            int idx = threadIdx.x + blockDim.x * blockIdx.x;
#            int idy = threadIdx.y + blockDim.y * blockIdx.y;
#            
#            float result = 0.0;
#            
#            for(int iter = 0; iter < %(NDIM)s; iter++) {
#                
#                float x_e = x[%(NDIM)s * idy + iter];
#                float y_e = y[%(NDIM)s * idx + iter];
#                
#                float denom = (fabs(x_e) + fabs(y_e));
#                
#                if (denom != 0) result += fabs(x_e - y_e) / denom;
#            }
#            int pos = idx + %(NCOLS)s * idy;
#            solution[pos] = result;
#        }
#    """
#    
#    kernel_code = kernel_code_template % {
#        'NCOLS': cols,
#        'NDIM': X.shape[1]
#    }
#    
#    mod = SourceModule(kernel_code)
#    
#    func = mod.get_function("canberra")
#    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(cols, rows, 1))
#    
#    return solution
#

def chebyshev(X, Y):
    raise NotImplementedError

def jaccard_coefficient(X, Y):
    raise NotImplementedError

def mahalanobis(X, Y):
    raise NotImplementedError

def braycurtis(X, Y):
    raise NotImplementedError

def sorensen_coefficient(X, Y):
    raise NotImplementedError

def spearman_coefficient(X, Y):
    raise NotImplementedError

def loglikehood_coefficient(X, Y):
    raise NotImplementedError    