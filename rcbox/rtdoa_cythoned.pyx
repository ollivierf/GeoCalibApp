# coding: utf-8

from __future__ import division, print_function
import numpy as np
from numpy.linalg import norm
from libc.math cimport copysign, abs

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef double _Slambda_c(double var,
                    double lbda): # st operator

    cdef double var_out = copysign(1.0, var) * max(abs(var) - lbda/2, 0.0)

    return var_out


#def _Slambda(np.ndarray[double, ndim=1, mode="c"] var,
#             double lbda): # Soft thresholding Operator
#    var_out = np.zeros_like(var)
#    _Slambda_c(var, var_out, lbda)
#    return var_out


cdef extern from "rtdoa.h":
    int MakeDsr_c(double *Dsr, double *s, double *r, int K, int L)


def MakeDsr(np.ndarray[double, ndim=2, mode="c"] Dsr,
             np.ndarray[double, ndim=2, mode="c"] s,
             np.ndarray[double, ndim=2, mode="c"] r,
             int K, int L):
    return MakeDsr_c(&Dsr[0,0], &s[0,0], &r[0,0], K, L)


cdef extern from "rtdoa.h":
    int UpdateEpsMumMun_c(double *eps, double *mum, double *mun,
                          double *Dsr, double *tau, double *t, double *o,
                          double c0, int K, int L)

def UpdateEpsMumMun(np.ndarray[double, ndim=3, mode="c"] eps,
                    np.ndarray[double, ndim=3, mode="c"] mum,
                    np.ndarray[double, ndim=3, mode="c"] mun,
                    np.ndarray[double, ndim=2, mode="c"] Dsr,
                    np.ndarray[double, ndim=3, mode="c"] tau,
                    np.ndarray[double, ndim=1, mode="c"] t,
                    np.ndarray[double, ndim=3, mode="c"] o,
                    double c0, int K, int L):
    return UpdateEpsMumMun_c(&eps[0,0,0], &mum[0,0,0], &mun[0,0,0],
                             &Dsr[0,0], &tau[0,0,0], &t[0], &o[0,0,0],
                             c0, K, L)


cdef extern from "rtdoa.h":
    int UpdateE_c(double *e_, double *s, double *r, double *Dsr, int K, int L)


def UpdateE(np.ndarray[double, ndim=3, mode="c"] e,
            np.ndarray[double, ndim=2, mode="c"] s,
            np.ndarray[double, ndim=2, mode="c"] r,
            np.ndarray[double, ndim=2, mode="c"] Dsr,
            int K, int L):
    return UpdateE_c(&e[0,0,0], &s[0,0], &r[0,0], &Dsr[0,0], K, L)


cdef extern from "rtdoa.h":
    int UpdateS_c(double *mum, double *e_, double *s, double *r, int K, int L)


def UpdateS(np.ndarray[double, ndim=3, mode="c"] mum,
            np.ndarray[double, ndim=3, mode="c"] e,
            np.ndarray[double, ndim=2, mode="c"] s,
            np.ndarray[double, ndim=2, mode="c"] r,
            int K, int L):
    return UpdateS_c(&mum[0,0,0], &e[0,0,0], &s[0,0], &r[0,0], K, L)


cdef extern from "rtdoa.h":
    int UpdateR_c(double *mun, double *e_, double *r, double *s, int K, int L)


def UpdateR(np.ndarray[double, ndim=3, mode="c"] mun,
            np.ndarray[double, ndim=3, mode="c"] e,
            np.ndarray[double, ndim=2, mode="c"] r,
            np.ndarray[double, ndim=2, mode="c"] s,
            int K, int L):
    return UpdateR_c(&mun[0,0,0], &e[0,0,0], &r[0,0], &s[0,0], K, L)


cdef extern from "rtdoa.h":
    int UpdateT_c(double *t, double *Dsr, double *mun, double c0, int K, int L)


def UpdateT(np.ndarray[double, ndim=1, mode="c"] t,
            np.ndarray[double, ndim=2, mode="c"] Dsr,
            np.ndarray[double, ndim=3, mode="c"] mun,
            double c0, int K, int L):
    return UpdateT_c(&t[0], &Dsr[0,0], &mun[0,0,0], c0, K, L)

cpdef UpdateO(np.ndarray[double, ndim=3, mode="c"] o,
            np.ndarray[double, ndim=3, mode="c"] tau,
            np.ndarray[double, ndim=2, mode="c"] Dsr,
            double c0, int K, int L, double param_l1):

    cdef int k, m, n
    cdef double val

    for k in xrange(K):
        for m in xrange(L):
                for n in xrange(L):
                    val = tau[k, m, n] - (Dsr[k, m] - Dsr[k, n])/c0
                    o[k, m, n] = _Slambda_c(val, param_l1)
    return 1

cdef extern from "rtdoa.h":
    int Step_c(double *Dsr, double *eps, double *mum, double *mun,
               double *e_, double *s, double *r, double *tau, double *t,
               double *o, double c0, int K, int L)

def Step(np.ndarray[double, ndim=2, mode="c"] Dsr,
         np.ndarray[double, ndim=3, mode="c"] eps,
         np.ndarray[double, ndim=3, mode="c"] mum,
         np.ndarray[double, ndim=3, mode="c"] mun,
         np.ndarray[double, ndim=3, mode="c"] e,
         np.ndarray[double, ndim=2, mode="c"] s,
         np.ndarray[double, ndim=2, mode="c"] r,
         np.ndarray[double, ndim=3, mode="c"] tau,
         np.ndarray[double, ndim=1, mode="c"] t,
         np.ndarray[double, ndim=3, mode="c"] o,
         double c0, int K, int L, double param_l1):
    Step_c(&Dsr[0,0], &eps[0,0,0], &mum[0,0,0], &mun[0,0,0],
           &e[0,0,0], &s[0,0], &r[0,0], &tau[0,0,0], &t[0], &o[0,0,0],
           c0, K, L)
    MakeDsr_c(&Dsr[0,0], &s[0,0], &r[0,0], K, L)
    UpdateO(o, tau, Dsr, c0, K, L, param_l1)



cpdef int Run_fast(int itmax, int verbose,
                 np.ndarray[double, ndim=3, mode="c"] r_iters,
                 np.ndarray[double, ndim=3, mode="c"] s_iters,
                 #np.ndarray[double, ndim=4, mode="c"] o_iters,
                 np.ndarray[double, ndim=2, mode="c"] Dsr,
                 np.ndarray[double, ndim=3, mode="c"] eps,
                 np.ndarray[double, ndim=3, mode="c"] mum,
                 np.ndarray[double, ndim=3, mode="c"] mun,
                 np.ndarray[double, ndim=3, mode="c"] e,
                 np.ndarray[double, ndim=2, mode="c"] s,
                 np.ndarray[double, ndim=2, mode="c"] r,
                 np.ndarray[double, ndim=3, mode="c"] tau,
                 np.ndarray[double, ndim=1, mode="c"] t,
                 np.ndarray[double, ndim=3, mode="c"] o,
                 double c0, int K, int L,
                 double param_l1, double param_converge):
                     
    cdef int it, l, k, j
    cdef int l_row, l_col
    cdef double crit_converge

    for it in xrange(itmax):
        for l in xrange(L):
            for j in xrange(3):
                r_iters[it, l, j] = r[l, j]

        for k in xrange(K):
            for j in xrange(3):
                s_iters[it, k, j] = s[k, j]

        #for k in xrange(K):
        #    for l_row in xrange(L):
        #        for l_col in xrange(L):
        #            o_iters[it, k, l_row, l_col] = o[k, l_row, l_col]

        Step_c(&Dsr[0,0], &eps[0,0,0], &mum[0,0,0], &mun[0,0,0],
               &e[0,0,0], &s[0,0], &r[0,0], &tau[0,0,0], &t[0], &o[0,0,0],
               c0, K, L) # Update geometry estimate
        # update outlier estimate:
        MakeDsr_c(&Dsr[0,0], &s[0,0], &r[0,0], K, L)
        UpdateO(o, tau, Dsr, c0, K, L, param_l1)

        # Test convergence
        crit_converge = norm(r - r_iters[it]) / norm(r)
        if crit_converge < param_converge:
            break

        if verbose:
            print("iteration %d/%d   f_cost = %.6f" % (it, itmax,
                                                        np.abs(eps).mean()))
            print("           crit_converge = %.6f" % (crit_converge))

    # Get last step
    for l in xrange(L):
        for j in xrange(3):
            r_iters[it+1, l, j] = r[l, j]

    for k in xrange(K):
        for j in xrange(3):
            s_iters[it+1, k, j] = s[k, j]

    #for k in xrange(K):
    #    for l_row in xrange(L):
    #        for l_col in xrange(L):
    #            o_iters[it+1, k, l_row, l_col] = o[k, l_row, l_col]
    
    return it
