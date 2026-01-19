# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 11:29:00 2016

@author: Charles Vanwynsberghe

"""
from __future__ import division, print_function

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform

def make_TOA(S, X, c0=340.):
    """
    Calculates TOA from sources (S) and microphones (X) positions.

    """

    Ns, Nr = S.shape[0], X.shape[0]
    TOA = np.zeros((Ns, Nr))

    for ns in range(Ns):
        for nr in range(Nr):
            TOA[ns, nr] = norm(S[ns] - X[nr]) / c0

    return TOA


def make_TDOA(S, X, c0=340.):
    """
    Generates TDOA from sources' (S) and microphones' (X) positions.
    """
    tdoa = np.zeros((S.shape[0], X.shape[0], X.shape[0]))

    for i, si in enumerate(S):
            for n, xn in enumerate(X):
                for m, xm in enumerate(X):
                    tdoa[i, n, m] = (norm(xn - si) - norm(xm - si)) / c0

    return tdoa


def get_D(X):
    """
    Returns distance matrix from X.

    """
    return squareform(pdist(X))


def makeWlocal(D, dmax):
    """
    Build weighting matrix for local MDS.

    Parameters
    ----------
    D : (Nr, Nr) array numpy
        Full Euclidean distance matrix
    dmax : float
        max distance (meters)

    Returns
    -------
    W : (Nr, Nr) array numpy
        Weighting matrix

    """
    W = np.zeros_like(D)
    W[np.where(D <= dmax)] = 1
    W[np.where(D > dmax)] = 0
    np.fill_diagonal(W, 0)
    return W
