# PycudaDistances

  Fast python distance computations in a variety of metrics that uses Graphics Processing Units with PyCuda.
  
  This is a framework that will allow general distance metrics to be incorporated into tree-based neighbors searches. 
  The idea is that we need a fast way to compute the distance between two points under a given metric. 

## Usage

	>>> from pycudadistances.distances import euclidean_distances
	>>> X = [[0, 1], [1, 1]]
    
    >>> euclidean_distances(X, X)
    array([[ 0.,  1.],
           [ 1.,  0.]])
    
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[ 1.        ],
           [ 1.41421356]])

## Bugs, Feedback

  Please submit bugs you might encounter, as well Patches and Features Requests to the [Issues Tracker](https://github.com/vinigracindo/pycudaDistances/issues) located at GitHub.

## Contributions

  If you want to submit a patch to this project, it is AWESOME. Follow this guide:
  
  * Fork PycudaDistances
  * Make your alterations and commit
  * Create a topic branch - git checkout -b my_branch
  * Push to your branch - git push origin my_branch
  * Create a [Pull Request](http://help.github.com/pull-requests/) from your branch.
  * You just contributed to the PycudaDistances project!


## Dependencies

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, PyCuda.

## Install

To install for all users on Unix/Linux::

    sudo python setup.py install

## LICENCE (GNU GPL)

Copyright (c) 2013

All rights reserved.
