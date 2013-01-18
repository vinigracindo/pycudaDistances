import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises

import distances
                        
def test_euclidean_distances():
    """Check that the pairwise euclidian distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = distances.euclidean_distances(X, X)
    assert_array_almost_equal(D, [[1.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = distances.euclidean_distances(X, X, inverse=False)
    assert_array_almost_equal(D, [[0.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError, distances.euclidean_distances, X, Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = distances.euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[0.29429806]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = distances.euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[0.29429806], [0.29429806]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = distances.euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[0.29429806, 1.], [0.29429806,  1.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = distances.euclidean_distances(X, X)
    assert_array_almost_equal(D, [[1., 0.29429806], [0.29429806, 1.]])

    X = [[1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0, 0.0]]
    D = distances.euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[0.5], [0.41421356]])
    
test_euclidean_distances()