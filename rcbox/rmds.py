# -*- coding: utf-8 -*-

"""
Module contenant toutes les fonctions pour
la calibration géométrique d'antenne

Created on Wed May 14 12:20:11 2014
@author: Charles Vanwynsberghe

"""
import numpy as np
from numpy import linalg
from scipy.spatial.distance import squareform, pdist


def _Slambda(var, lbda):
    """
    Soft thresholding Operator, elementwise on var.

    """
    out = np.sign(var) * np.maximum(np.abs(var) - lbda/2, 0.0)
    return out


def _e_vect(n, N):
    """
    Return n-th canonical basis vector.

    """
    out = np.zeros((1, N)).T
    out[n, 0] = 1
    return out


def _Lmake(N, W=None):
    """
    Compute matrix L (cf. paper)

    """
    L = np.zeros((N, N))
    if W is None:
            L[:, :] = -1
            L += N*np.eye(N)
    else:
        for nn in range(N):
            for mm in range(nn, N):
                L += W[nn, mm] * np.dot(_e_vect(nn, N) - _e_vect(mm, N),
                                        (_e_vect(nn, N) - _e_vect(mm, N)).T)
    return L


def compute_Lpinv(Nr, W):
    """
    Compute pseudo-inverse of L matrix (left matrix in Guttman transform)

    """
    L = _Lmake(Nr, W)
    Lpinv = linalg.pinv(L).copy(order="C")

    return Lpinv


def rmdsw(D, lbda=0.5, Ndim=3, W=None, Xinit=None,
          Maxit=5000, EpsLim=10**-6, EpsType="Forero2012", verbose=1):
    """
    Weighted Robust MDS (WRMDS). Weighted version of RMDS in Forero2012 [1]_.

    Parameters
    ----------
    D : (Nr, Nr) numpy array
        Full Euclidean distance matrix

    lbda : float
        Regularization parameter for finding outliers.

    Ndim : int
        Dimension of projection space (output space)

    W : (Nr, Nr) numpy array
        Weighting matrix

    Xinit : (Nr, Ndim) numpy array
        Coordinates X at initial algorithm step

    Maxit : int
        Maximum number of iterations

    EpsLim : float
        Stopping criterion for algorithm

    EpsType : "str"
        Criterion type :

            - `"Forero2012"`

            - `"meters"`

    Returns
    -------
    X : (Nr, Ndim) numpy array
        Matrix of coordinates in Ndim space

    O : (Nr, Nr) numpy array
        Matrix of outlying errors

    Eps : numpy array
        Criterion values at each algorithm step

    References
    ----------
    .. [1] Sparsity-Exploiting Robust Multidimensional Scaling.

    """
    Nr = D.shape[0]

    if W is None:  # if W is None -> wij = 1 for all (i,j)
        W = np.ones((Nr, Nr))
        np.fill_diagonal(W, 0)

    Wflat = squareform(W)
    Dflat = squareform(D)

    X = np.zeros((Maxit, Ndim, Nr))
    if Xinit is None:
        X[0, :, :] = np.random.randn(Ndim, Nr)
    else:
        X[0, :, :] = Xinit.T

    O = np.zeros((Maxit, int(Nr*(Nr-1)/2)))

    Lpinv = compute_Lpinv(Nr, W)

    A1 = np.zeros((Nr, Nr))
    A11 = np.zeros((int(Nr*(Nr-1)/2),))
    L1 = np.zeros((Nr, Nr))

    Eps = np.zeros(Maxit,)
    Err = np.zeros_like(Eps)

    for t in range(Maxit-1):
        if verbose: print('t: %d , Eps(t-1): %.7f' %(t, Eps[t-1]))
        DDt = pdist(X[t, :, :].T)

        # Update O
        for nm in range(Nr*(Nr-1)//2):
                O[t+1, nm] = _Slambda(Wflat[nm]*(Dflat[nm] - DDt[nm]), lbda)

        # Compute L1(O(t+1), X(t))
        for nm in range(Nr*(Nr-1)//2):
                if DDt[nm] != 0 and Dflat[nm] > O[t+1, nm]:
                    A11[nm] = Wflat[nm] * (Dflat[nm]-O[t+1, nm])/DDt[nm]
                else:
                    A11[nm] = 0

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
    X = X[0:t+1, :, :].transpose(0, 2, 1)
    O = O[0:t+1, :]

    return X, O, Eps


class RMDS:
    """
    Class for Weighted Robust MDS [1]_.

    References
    ----------
    .. [1] Sparsity-Exploiting Robust Multidimensional Scaling.

    """
    def __init__(self, D=None):
        """
        Parameters
        ----------
        D: (M, M) numpy array
            Distance matrix

        """

        self.d = 3
        self.it = 0

        self.D = D
        self.Nr = self.D.shape[0]
        self.it = 0

    def Run(self, lbda, W=None, Xinit=None, verbose=0, itmax=1000,
            EpsLim=1e-6, EpsType="Forero2012"):
        """
        Parameters
        ----------
        lbda : float
            Regularization parameter for finding outliers.

        W : (Nr, Nr) numpy array
            Weighting matrix

        Xinit : (Nr, Ndim) numpy array
            Coordinates X at initial algorithm step

        verbose : boolean
            more print

        itmax : int
            Maximum number of iterations

        EpsLim : float
            Stopping criterion for algorithm

        EpsType : "str"
            Criterion type :

                - `"Forero2012"`

                - `"meters"`

        """

        self.itmax = itmax

        if W is None:  # if W is None -> wij = 1 for all (i,j)
            self.W = np.ones((self.Nr, self.Nr))
            np.fill_diagonal(self.W, 0)

        else:
            self.W = W

        from . import rmds_cythoned
        res = rmds_cythoned.rmdsw(self.D, lbda=lbda, Ndim=self.d, W=self.W,
                                  Xinit=Xinit, Maxit=self.itmax,
                                  EpsLim=EpsLim,
                                  EpsType=EpsType, verbose=verbose)

        self.X = res[0]
        self.O = res[1]
        self.Eps = res[2]
        self.it = self.Eps.shape[0]  # total number of iterations

    def align(self, x_ref, which="last"):
        """
        Align with Kabsch algorithm last or all iterations in self.X

        Parameters
        ----------
        x_ref : (Nr, d) numpy array
            Reference shape for alignment.

        which : "last" or (Nr, d) numpy array
            Shape to rotate.

        """
        self.x_ref = x_ref
        self.centroid_ref = x_ref.mean(axis=0)
        self.x_ref_centered = x_ref - self.centroid_ref

        if which == "last":
            x_to_align = self.X[-1, ...]
        else:
            x_to_align = which.copy()

        centroid_to_align = x_to_align.mean(axis=0)
        x_to_align_centered = x_to_align - centroid_to_align

        cov = x_to_align_centered.T @ self.x_ref_centered

        self.u, self.s, self.v = np.linalg.svd(cov)

        e = np.eye(self.d)
        e[-1, -1] = np.sign(self.s.prod())

        # create Rotation matrix U
        self.r = self.v.T @ e @ self.u.T

        # apply rotation
        self.x_aligned = x_to_align_centered @ self.r

    def tune_lambda(self, lambda_sequence, lambda_init=1e6,
                    W=None, Xinit=None, verbose=0, itmax=1000,
                    EpsLim=1e-6, EpsType="Forero2012"):
        """
        Empirical lambda tune by L curve

        Parameters
        ----------
        lambda_sequence : (n_seq,) numpy array
            Sequence of lambda values

        other parameters : cf. self.Run(...)

        """
        self.lambda_sequence = lambda_sequence

        self.n_seq = self.lambda_sequence.size
        self.X_seq = []
        self.O_seq = []
        self.Eps_seq = []
        self.it_seq = []
        self.k_seq = []

        # pre warm initial guess
        self.Run(lambda_init, W=W, Xinit=Xinit,
                 verbose=verbose, itmax=itmax,
                 EpsLim=EpsLim, EpsType=EpsType)
        Xinit_up = self.X[-1, ...].copy()

        for n_, lambda_val in enumerate(self.lambda_sequence):

            self.Run(lambda_val, W=W, Xinit=Xinit_up, verbose=verbose,
                     itmax=itmax, EpsLim=EpsLim, EpsType=EpsType)

            Xinit_up = self.X[-1, ...].copy()

            self.X_seq.append(self.X)
            self.O_seq.append(self.O)
            self.Eps_seq.append(self.Eps)
            self.it_seq.append(self.Eps.shape[0])
            self.k_seq.append((self.O[-1, ...] != 0).sum())

        self.k_seq = np.array(self.k_seq)


def compute_Linv_mdu(Nr, Ns):
    """
    Fast implementation of Lpinv - only in case of MDU weight matrix.

    """
    N = Nr + Ns
    A = np.ones((N,))
    A[0:Nr] = Ns
    A[Nr::] = Nr
    A_inv = np.diag(A**-1)

    u = np.zeros((N, 1))
    u[Nr::] = 1
    v = np.zeros((N, 1))
    v[0:Nr] = 1
    U = np.hstack((u, v))
    V = np.vstack((v.T, u.T))

    norm_rs = (1 + Nr**2 * Ns**-2)**0.5
    norm_sr = (1 + Ns**2 * Nr**-2)**0.5
    D_inv_bis = (np.array([[1/norm_rs], [-Nr*Ns**-1/norm_rs]]) @
                 np.array([[-1/norm_sr, Ns*Nr**-1/norm_sr]]))
    D_inv_bis /= (norm_rs*norm_sr)

    L_inv = A_inv - A_inv @ U @ D_inv_bis @ V @ A_inv

    return L_inv


class RMDU(RMDS):
    """
    Class for Robust Multidimensional Unfolding.

    """
    def __init__(self, D_block=None, fast_Linv=True):
        """
        Inputs
        ------
        D_block: (Ns, Nr) numpy array
            Distance matrix block. D_block = c0 * TOA

        """

        self.d = 3
        self.it = 0

        self.Ns = D_block.shape[0]
        self.Nr = D_block.shape[1]
        self.N = self.Nr + self.Ns

        self.D_block = D_block

        self.D = np.zeros((self.N, self.N))
        self.D[self.Nr::, 0:self.Nr] = self.D_block
        self.D = self.D + self.D.T

        self.W = np.zeros((self.N, self.N))
        self.W[self.Nr::, 0:self.Nr] = 1.
        self.W = self.W + self.W.T

        if fast_Linv is True:
            self.Lpinv = compute_Linv_mdu(self.Nr, self.Ns)

        else:
            self.Lpinv = compute_Lpinv(self.N, self.W)

        self.Xinit = None

    def Run(self, lbda, Xinit=None, verbose=0, itmax=1000,
            EpsLim=1e-6, EpsType="Forero2012"):
        """
        Parameters
        ----------
        lbda : float
            Regularization parameter for finding outliers.

        Xinit : (Nr, Ndim) numpy array
            Coordinates X at initial algorithm step

        verbose : boolean
            more print

        itmax : int
            Maximum number of iterations

        EpsLim : float
            Stopping criterion for algorithm

        EpsType : "str"
            Criterion type :

                - `"Forero2012"`

                - `"meters"`

        """
        self.itmax = itmax

        from . import rmds_cythoned
        
        res = rmds_cythoned.rmdsw(self.D, lbda=lbda, Ndim=self.d, W=self.W,
                                  Xinit=Xinit, Maxit=self.itmax,
                                  EpsLim=EpsLim,
                                  EpsType=EpsType, verbose=verbose,
                                  Lpinv_precomputed=self.Lpinv)

        self.X = res[0]
        self.O = res[1]
        self.Eps = res[2]
        self.it = self.Eps.shape[0]  # total number of iterations

    def tune_lambda(self, lambda_sequence, lambda_init=1e6,
                    Xinit=None, verbose=0, itmax=1000,
                    EpsLim=1e-6, EpsType="Forero2012"):
        """
        Empirical lambda tune by L curve

        Parameters
        ----------
        lambda_sequence : (n_seq,) numpy array
            Sequence of lambda values

        other parameters : cf. self.Run(...)

        """
        self.lambda_sequence = lambda_sequence

        self.n_seq = self.lambda_sequence.size
        self.X_seq = []
        self.O_seq = []
        self.Eps_seq = []
        self.it_seq = []
        self.k_seq = []

        # pre warm initial guess
        self.Run(lambda_init, Xinit=Xinit,
                 verbose=verbose, itmax=itmax,
                 EpsLim=EpsLim, EpsType=EpsType)

        Xinit_up = self.X[-1, ...].copy()

        for n_, lambda_val in enumerate(self.lambda_sequence):

            self.Run(lambda_val, Xinit=Xinit_up, verbose=verbose,
                     itmax=itmax, EpsLim=EpsLim, EpsType=EpsType)

            Xinit_up = self.X[-1, ...].copy()

            self.X_seq.append(self.X)
            self.O_seq.append(self.O)
            self.Eps_seq.append(self.Eps)
            self.it_seq.append(self.Eps.shape[0])
            self.k_seq.append((self.O[-1, ...] != 0).sum())

        self.k_seq = np.array(self.k_seq)
