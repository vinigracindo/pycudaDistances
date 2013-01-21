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

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --home


To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

## LICENCE (GNU GPL)

Copyright (c) 2013

All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MURIÇOCA LABS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
