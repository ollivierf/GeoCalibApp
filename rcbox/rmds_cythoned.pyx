# -*- coding: utf-8 -*-

"""
Module contenant toutes les fonctions pour
la calibration géométrique d'antenne

Created on Wed May 14 12:20:11 2014
@author: Charles Vanwynsberghe

"""

from __future__ import division
import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport copysign, abs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

import scipy as sp

from scipy.spatial.distance import squareform, pdist
from scipy import linalg as linalg

from .rmds import _e_vect, _Lmake, compute_Lpinv


cdef int _Slambda_c(np.ndarray[double, ndim=1, mode="c"] var,
                    np.ndarray[double, ndim=1, mode="c"] var_out,
                    double lbda):

    cdef int k
    cdef int var_size = var.size

    for k in range(var_size):
        var_out[k] = copysign(1.0, var[k]) * max(abs(var[k]) - lbda/2, 0.0)

    return 1


def _Slambda(np.ndarray[double, ndim=1, mode="c"] var,
             double lbda): # Soft thresholding Operator
    var_out = np.zeros_like(var)
    _Slambda_c(var, var_out, lbda)
    return var_out


def rmdsw(D, lbda=0.5, Ndim=3, W=None, Xinit=None,
          Maxit=5000, EpsLim=10**-6, EpsType="Forero2012",
          verbose=1, Lpinv_precomputed=None):

    cdef int Nr = D.shape[0]
    cdef int nm, t

    if W is None:  # if W is None -> wij = 1 for all (i,j)
        W = np.ones((Nr,Nr))
        np.fill_diagonal(W, 0)

    cdef np.ndarray[double, ndim=1, mode="c"] Wflat = squareform(W)
    cdef np.ndarray[double, ndim=1, mode="c"] Dflat = squareform(D)

    cdef np.ndarray[double, ndim=3, mode="c"] X = np.zeros((Maxit, Ndim, Nr))
    if Xinit is None:
        X[0, :, :] = np.random.randn(Ndim, Nr)
    else:
        X[0, :, :] = Xinit.T

    cdef np.ndarray[double, ndim=2, mode="c"] O = np.zeros((Maxit, int(Nr*(Nr-1)/2)))

    cdef np.ndarray[double, ndim=2, mode="c"] Lpinv
    if Lpinv_precomputed is None:
        Lpinv = compute_Lpinv(Nr, W)
    else:
        Lpinv = Lpinv_precomputed

    cdef np.ndarray[double, ndim=1, mode="c"] DDt = np.zeros((int(Nr*(Nr-1)//2),))
    cdef np.ndarray[double, ndim=2, mode="c"] A1 = np.zeros((Nr, Nr))
    cdef np.ndarray[double, ndim=1, mode="c"] A11 = np.zeros((int(Nr*(Nr-1)/2),))
    cdef np.ndarray[double, ndim=2, mode="c"] L1 = np.zeros((Nr, Nr))

    cdef np.ndarray[double, ndim=1, mode="c"] Eps = np.zeros(Maxit,)
    cdef np.ndarray[double, ndim=1, mode="c"] Err = np.zeros_like(Eps)

    for t in xrange(Maxit-1):
        if verbose: print('t: %d , Eps(t-1): %.7f' %(t, Eps[t-1]))
        DDt = pdist(X[t, :, :].T)
        
        # Update O
        _Slambda_c(Wflat*(Dflat - DDt), O[t+1, :], lbda)

        # Compute L1(O(t+1), X(t))
        for nm in xrange(Nr*(Nr-1)//2):
                if DDt[nm] != 0 and Dflat[nm] > O[t+1, nm]:
                    A11[nm] = Wflat[nm] * (Dflat[nm]-O[t+1, nm])/DDt[nm]
                else:
                    A11[nm] = 0.0

        A1 = squareform(A11)
        L1 = np.diag(A1.sum(1)) - A1

        # Update X
        X[t+1, :, :] = np.dot(X[t, :, :], np.dot(L1, Lpinv))
        if EpsType == "Forero2012":
            Eps[t] = linalg.norm(X[t+1, :, :]-X[t, :, :]) / linalg.norm(X[t+1, :, :])
        elif EpsType == "meters":
            Eps[t] = linalg.norm(X[t+1, :, :] - X[t, :, :])

        # Stopping condition
        if Eps[t] < EpsLim:
            break

    Err = Err[0:t+1]
    Eps = Eps[0:t+1]
    X = X[0:t+1, :, :].transpose(0, 2, 1).copy(order="C")
    O = O[0:t+1, :]

    return X, O, Eps
