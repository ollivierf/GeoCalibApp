# -*- coding: utf-8 -*-
"""
Created on Fri Jul 8 2016

@author: charles

Solve the tdoa microphone calibration problem with sparse outliers in tdoa set
"""
import itertools

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GeoCalibUtils import fit_line_and_adjust_spacing_8, procrustes

def _Slambda(var, lbda):
    """
    Soft thresholding Operator, elementwise on var.

    """
    out = np.sign(var) * np.maximum(np.abs(var) - lbda/2, 0.0)
    return out


class ROno_py():
    """
    A robust tdoa geometric calibration.

    Algorithm alternates Ono2009 subproblem with Lasso problem on outliers.

    """

    def __init__(self, c0=340, tau=None):
        """
        Inputs
        ------
        c0: float
            Sound speed, m/s.

        tau: (K, L, L) numpy array
            TDOA matrix, for L microphones and K different sources

        """
        self.c0 = c0
        self.D = 3
        self.itmax = 1000
        self.it = 0

        if tau is not None:
            self.SetTDOA(tau)

    def SetTDOA(self, tau):
        """
        Set tdoa matrix and corresponding attributes.

        """
        assert tau.ndim == 3, u"TDOA matrix must be (Ns, M, M)"
        assert tau.shape[1] == tau.shape[2], u"TDOA matric must be (Ns, M, M)"

        self.tau = tau
        self.K = tau.shape[0]
        self.L = tau.shape[1]

        assert (self.K-4)*(self.L-4) >= 9, "wrong L & K"

    def Init(self, r0=None, s0=None):
        """
        Inputs
        ------
        r0: (L, D) numpy array
            Microphone positions for initial step.

        s0: (K, D) numpy array
            Source positions for initial step.

        """
        # Condition initiale r et s
        if r0 is not None:
            self.r0 = r0
        else:
            self.r0 = np.random.randn(self.L, self.D)

        if s0 is not None:
            self.s0 = s0
        else:
            self.s0 = np.random.randn(self.K, self.D)

        # Init variables
        self.r = self.r0.copy()
        self.s = self.s0.copy()

        self.eps = np.zeros_like(self.tau)
        self.mum = np.zeros_like(self.tau)
        self.mun = np.zeros_like(self.tau)
        self.e = np.zeros((self.K, self.L, self.D))
        self.t = np.zeros((self.L,))
        self.Dsr = np.zeros((self.K, self.L))
        self.o = np.zeros_like(self.tau)

    def MakeDsr(self):
        """
        Compute matrix of source-microphone distances

        """
        for k in range(self.K):
            for l in range(self.L):
                self.Dsr[k, l] = norm(self.s[k] - self.r[l])

    def UpdateEpsMumMun(self):
        """
        Updates eps, mu^m and mu^n. Eqs (32), (33), (34)

        """
        for k, m, n in itertools.product(range(self.K), range(self.L),
                                         range(self.L)):
            self.eps[k, m, n] = self.Dsr[k, m] - self.Dsr[k, n] - self.c0 * \
                                (self.tau[k, m, n] - self.o[k, m, n] +
                                 self.t[m] - self.t[n])
            self.mum[k, m, n] = self.Dsr[k, m] - 0.5*self.eps[k, m, n]
            self.mun[k, m, n] = self.Dsr[k, n] + 0.5*self.eps[k, m, n]

    def UpdateE(self):
        """
        Updates e. Eq (35)

        """
        for k in range(self.K):
            for n in range(self.L):
                self.e[k, n] = (self.s[k] - self.r[n])/self.Dsr[k, n]

    def UpdateS(self):
        """
        Updates s. Eq (36)

        """
        for i in range(self.K):
            tot = np.zeros((self.D,))
            for m in range(self.L):
                tot += self.L*self.r[m] + self.e[i, m]*np.sum(self.mum[i, m, :])
            self.s[i] = (1/self.L**2)*tot

    def UpdateR(self):
        """
        Updates r. Eq (37)

        """
        for n in range(self.L):
            tot = np.zeros((self.D,))
            for i in range(self.K):
                tot += self.L*self.s[i] - self.e[i, n]*np.sum(self.mun[i, :, n]) # !!!Erreur dans le proceedings eq (37); inversion nm<>mn
            self.r[n] = (1/(self.K*self.L)) * tot

    def UpdateT(self):
        """
        Updates t. Eq (38)

        """
        for n in range(self.L):
            tot = 0
            for i in range(self.K):
                tot += self.L*self.Dsr[i, n] - np.sum(self.mun[i, :, n])
            self.t[n] += 1/(self.c0*self.K*self.L)*tot

    def UpdateO(self):
        """
        Update outlying error terms

        """
        for k in range(self.K):
            for m in range(self.L):
                    for n in range(self.L):
                        val = self.tau[k, m, n] - \
                              (self.Dsr[k, m] - self.Dsr[k, n])/self.c0
                        self.o[k, m, n] = _Slambda(val, self.param_l1)

    def Step(self):
        """
        Do one iteration (all update rules). Eqs (32)-(38)

        """
        self.MakeDsr()
        self.UpdateEpsMumMun()
        self.UpdateE()

        self.UpdateS()
        self.UpdateR()

        self.MakeDsr()
        self.UpdateO()
        self.UpdateT()
        

    def Has_converged(self):
        """
        Test convergence of solution, between it and it+1 iterations.

        """
        self.crit_converge = norm(self.r - self.r_iters[self.it])/norm(self.r)

        return (self.crit_converge < self.param_converge)

    def Run(self, param_l1, itmax=1000, param_converge=1e-6, verbose=1, plot=False):
        """
        Run algorithm, for itmax iterations, or when convergence occurs.
        """
        self.itmax = itmax
        self.param_converge = param_converge
        self.param_l1 = param_l1

        # Get r of each iteration:
        self.r_iters = np.zeros((itmax + 1, self.L, self.D))
        # Get s of each iteration
        self.s_iters = np.zeros((itmax + 1, self.K, self.D))

        if plot:
            plt.ion()  # Enable interactive mode
            #plt.close('all')  # Close any existing figures
            # Initialize the figure
            fig = plt.figure(200)
            ax1 = fig.add_subplot(221)
            cost_plot, = ax1.plot([], [], 'b.')
            ax1.set_title('Cost function')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('cost function')
            ax1.set_xlim([0, itmax])
            ax1.set_ylim([-0.10, 1])

            ax2 = fig.add_subplot(223)
            convergence_plot, = ax2.plot([], [], 'b.')
            ax2.set_title('Convergence criterion')
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('Convergence criterion')
            ax2.set_xlim([0, itmax])
            ax2.set_ylim([-0.1, 0.1])

            ax3 = fig.add_subplot(122, projection='3d')
            estimated_plot = ax3.scatter(
                self.r[:, 0], self.r[:, 1], self.r[:, 2], marker='o', cmap='inferno',
                c=np.arange(len(self.r)), s=35, linewidths=1.5, label='r estimated'
            )
            r0 = np.load('C:/Users/froll/Documents/Paris6/Enseignement/M1/Modules/Projet/2425/Frogs/ManipAnech_250313/250320/r0.npy')
            r0 -= np.mean(r0, axis = 0)
            r0[24:,:] = r0[31:23:-1,:]

            measured_plot = ax3.scatter(
                    r0[:, 0], r0[:, 1], r0[:, 2], marker='x',edgecolors='k', cmap='inferno',
                    c=np.arange(len(self.r)), s=35, linewidths=1.5, label='r estimated'
                )
            ax3.set_xlim([-2, 2])
            ax3.set_ylim([-2, 2])
            ax3.set_zlim([-2, 2])
            ax3.set_aspect('equal')
            fig.tight_layout()

        cc = []  # Convergence criterion values
        cf = []  # Cost function values
        for self.it in range(self.itmax):
            self.r_iters[self.it] = self.r
            self.s_iters[self.it] = self.s

            # Perform one iteration
            self.Step()
            #  self.r[l*8:l*8+8, :] = fit_line_and_adjust_spacing_8(self.r[l*8:l*8+8, :],0.06)
            # r0 -= np.mean(r0, axis = 0)
            # Y = self.r
            # R = procrustes(r0,Y)
            # self.r = np.dot(self.r,R)
            
            # Check for convergence
            if self.Has_converged():
                break

            # Collect convergence and cost function values
            if verbose:
                print(f"iteration {self.it+1}/{self.itmax}   f_cost = {np.abs(self.eps).mean():.6f}  crit_conv = {self.crit_converge:.6f}\r", end='')
                cc.append(self.crit_converge)
                cf.append(np.abs(self.eps).mean())

            # Update the plots every 10 iterations
            if plot:# and self.it % 10 == 0:
                # Update convergence plot
                convergence_plot.set_xdata(np.arange(self.it + 1))
                convergence_plot.set_ydata(np.array(cc))

                # Update cost plot
                cost_plot.set_xdata(np.arange(self.it + 1))
                cost_plot.set_ydata(np.array(cf))

                # Remove the old scatter plot
                estimated_plot.remove()
                measured_plot.remove()
                # Create a new scatter plot with updated data
                estimated_plot = ax3.scatter(
                    self.r[:, 0], self.r[:, 1], self.r[:, 2], marker='o',edgecolors='k', cmap='inferno',
                    c=np.arange(len(self.r)), s=35, linewidths=1.5, label='r estimated'
                )
                
                measured_plot = ax3.scatter(
                    r0[:, 0], r0[:, 1], r0[:, 2], marker='x',edgecolors='k', cmap='inferno',
                    c=np.arange(len(self.r)), s=35, linewidths=1.5, label='r estimated'
                )
                ax3.set_xlim([-2, 2])
                ax3.set_ylim([-2, 2])
                ax3.set_zlim([-2, 2])
                ax3.set_aspect('equal')

                # Redraw the figure
                fig.canvas.draw()
                plt.pause(0.1)  # Allow the GUI event loop to process events

        # Get last step:
        self.r_iters[self.it + 1] = self.r
        self.s_iters[self.it + 1] = self.s

        # Truncate *_iters to last iteration if convergence:
        self.r_iters = self.r_iters[:self.it + 2]
        self.s_iters = self.s_iters[:self.it + 2]

        #print("self.r shape:", self.r.shape)
        #print("self.r data:", self.r)

    def Plot_result(self, r_gt=None, show_init=False,
                    show_iters=False, **kwargs):
        """
        Plot microphone positions.

        Inputs
        ------
        r_gt: (L, D) numpy array, optional
            Grount truth microphone positions.

        show_init: bool
            Plot initial guess microphone positions.

        show_iters: bool
            Plot all microphone positions updates.

        Returns
        -------
        fig: matplotlib figure

        ax: matplotlib object

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else:
            fig = plt.figure(num=kwargs.get('num', None))
            ax = fig.add_subplot(111, projection='3d')

        # Plot final
        ax.scatter(*self.r.T, marker='o', facecolor='b',
                   color=np.arange(len(self.r)),
                   s=35, linewidths=1.5, label='r estimated')

        # Plot initial guess
        if show_init:
            ax.scatter(*self.r_iters[0, :, :].T, marker='s', color='RoyalBlue',
                       facecolor='none',
                       s=35, linewidths=1.5, label='r inital')

        # Plot microphone paths across iterations
        if show_iters:
            for l in range(self.L):
                ax.plot(*self.r_iters[:, l, :].T, color='RoyalBlue', alpha=0.2)

        # Plot ground truth if given
        if r_gt is not None:
            ax.scatter(*r_gt.T, marker='o', color='red', facecolor='none',
                       s=35, linewidths=1.5, label='r ground truth')

            for l in range(self.L):
                ax.plot([r_gt[l, 0], self.r[l, 0]],
                        [r_gt[l, 1], self.r[l, 1]],
                        [r_gt[l, 2], self.r[l, 2]],
                        ':r', linewidth=1)
        ax.legend()

        if 'ax' not in kwargs:
            return fig, ax

    def __str__(self):
        string = "L = %d (Nb microphones) \n" % (self.L)
        string += "K = %d (Nb sources) \n" % (self.K)
        string += "D = %d (Ndim) \n" % (self.D)
        string += "c0 = %.1f m/s \n" % (self.c0)
        string += "it/itmax = %d/%d \n" % (self.it, self.itmax)

        return string

    def saveh5(self, fname):
        """
        Save results in fname file, hdf5 format.

        """
        import h5py
        from textwrap import dedent

        with h5py.File(fname, "w-") as f:
            readme = u"""\
                Geometric calibration using Robust TDOA approach

                K: nb sources
                L: nb microphones
                D: spatial dimension
                c0: sound speed
                TDOA: tdoa matrices

                r0, s0: microphone & source positions at initialisation
                r, s: microphone & source positions at end of calibration (estimated)
                o:  outlier tensor at the end of calibration (estimated)
                t:  internal channel delays (estimated)
                r_iters, s_iters: microphone & source positions at each iteration (updates)
                o_iters: outlier tensor at each iteration (updates)

                lambda: l1 regularization parameter for lasso problem on o
                it: number of iterations
                """
            f.create_dataset("readme", data=dedent(readme))
            f.create_dataset("TDOA", data=self.tau)
            f.create_dataset("r0", data=self.r0)
            f.create_dataset("s0", data=self.s0)
            f.create_dataset("r", data=self.r)
            f.create_dataset("s", data=self.s)
            f.create_dataset("o", data=self.o)
            f.create_dataset("t", data=self.t)
            f.create_dataset("r_iters", data=self.r_iters)
            f.create_dataset("s_iters", data=self.s_iters)
            #f.create_dataset("o_iters", data=self.o_iters)
            f.create_dataset("c0", data=self.c0)
            f.create_dataset("L", data=self.L)
            f.create_dataset("K", data=self.K)
            f.create_dataset("D", data=self.D)
            f.create_dataset("it", data=self.it+1)
            f.create_dataset("lambda", data=self.param_l1)


class ROno_cy(ROno_py):
    """
    The fast implementation of ROno (C + Cython)
    """

    def Run(self, param_l1, itmax=1000, param_converge=1e-6, verbose=1):
        """
        Run algorithm, for itmax iterations or when convergence occurs.

        """
        from .rtdoa_cythoned import Run_fast

        self.itmax = itmax
        self.param_converge = param_converge
        self.param_l1 = param_l1
        # Get r of each iteration:
        self.r_iters = np.zeros((itmax + 1, self.L, self.D))
        # Get s of each iteration:
        self.s_iters = np.zeros((itmax + 1, self.K, self.D))
        # Get o of each iteration
        #self.o_iters = np.zeros((itmax + 1, self.K, self.L, self.L))

        self.it = Run_fast(self.itmax, verbose,
                           self.r_iters, self.s_iters,# self.o_iters,
                           self.Dsr, self.eps,
                           self.mum, self.mun, self.e, self.s, self.r,
                           self.tau, self.t, self.o, self.c0,
                           self.K, self.L, self.param_l1, self.param_converge)

        # Truncate *_iters to last iteration if convergence:
        self.r_iters = self.r_iters[0:self.it+2]
        self.s_iters = self.s_iters[0:self.it+2]
        #self.o_iters = self.o_iters[0:self.it+2]


def ROno(c0=340., tau=None, fast=False):
    """
    Returns RTDOA calibration class, with python or C/Cython implementation

    """
    if fast is False:
        return ROno_py(c0, tau)

    elif fast is True:
        return ROno_cy(c0, tau)

# %% TDOA Denoising

def Pl(X, l):
    """
    Hard thresholding operator, projects X to its l-sparse approximation

    """
    Y = np.zeros_like(X)
    X_abs = np.abs(X)

    ids_max = np.argsort(X_abs, axis=None)[::-1]
    ids_max = np.unravel_index(ids_max, X.shape)
    id_l = [ids_max[0][0:l], ids_max[1][0:l]]

    Y[id_l] = X[id_l]
    return Y


def denoise_tdoa_hard(M_tilde, k, eps=1e-6, tmax=100, checks=1):
    """
    Denoise full TDOA matrix (outliers) [1]_ by matrix decomposition
    M_tilde = M + S (+ E).

    Parameters
    ----------
    M_tilde : (Nr, Nr) float numpy array
        TDOA matrix.

    k : unsigned int
        2k is maximum number of outliers supposed to be present in the TDOA matrix.

    eps : unsigned float
        stopping criterion. cf Algorithm 1 [1]_.

    tmax : unsigned int
        maximum iteration number.

    Returns
    -------
    Mt : (Nr, Nr) float numpy array
        denoised TDOA matrix.

    St : (Nr, Nr) float numpy array
        matrix of outlying errors.

    epst : unsigned float
        stopping criterion at last iteration.

    References
    ----------
    .. [1] TDOA Matrices: Algebraic Properties and their Application to
    Robust Denoising with Missing Data

    """
    Nr = M_tilde.shape[0]
    vone = np.ones((Nr, 1)) / (Nr**0.5)

    # Step 1: initialize
    M0, S0, t = M_tilde.copy(), np.zeros_like(M_tilde), 0
    Mt, St = M0.copy(), S0.copy()

    # Step 2 : convergence criterion
    converged = norm(M_tilde - Mt - St) / norm(M_tilde) < eps
    while ((converged) or t == 0) and t != tmax:
#        print(norm(M_tilde - Mt - St)/norm(M_tilde))

        t += 1
        Stm1 = St.copy()
        Mt = np.dot(np.dot(M_tilde - Stm1, vone), vone.T) \
            + np.dot(np.dot(vone, vone.T), M_tilde - Stm1)
        St = Pl(M_tilde - Mt, 2*k)

        if checks:
            np.testing.assert_allclose(St, -St.T, atol=1e-2, verbose=1)
            np.testing.assert_allclose(Mt, -Mt.T, atol=1e-2, verbose=1)
            assert np.count_nonzero(St) <= 2*k

    return Mt, St, norm(M_tilde - Mt - St) / norm(M_tilde)


def denoise_tdoa_soft(M_tilde, lbda, eps=1e-6, tmax=100, verbose=0, checks=1):
    """
    Denoise full TDOA matrix (outliers) [1]_ by matrix decomposition
    M_tilde = M + S (+ E).

    Soft thresholding version.

    Inputs
    ------

    M_tilde : (Nr, Nr) float numpy array
        TDOA matrix.

    lbda : float
        threshold for soft thresholding to build outlier matrix S.

    eps : unsigned float
        stopping criterion: convergence, NB: different from Algorithm 1 [1]_.

    tmax : unsigned int
        maximum iteration number.

    Returns
    -------
    Mt : (Nr, Nr) float numpy array
        denoised TDOA matrix.

    St : (Nr, Nr) float numpy array
        matrix of outlying errors.

    epst : unsigned float
        stopping criterion at last iteration.

    References
    ----------
    .. [1] TDOA Matrices: Algebraic Properties and their Application to
    Robust Denoising with Missing Data

    """

    if verbose:
        print("Start Semi-soft Denoising, lambda={}".format(lbda))

    Nr = M_tilde.shape[0]
    vone = np.ones((Nr, 1)) / (Nr**0.5)

    # Step 0: initialize
    M0, S0, t = M_tilde.copy(), np.zeros_like(M_tilde), 0
    Mt, St = M0.copy(), S0.copy()

    converged = False
    while (not(converged) or t == 0) and (t != tmax):

        t += 1
        Stm1 = St.copy()
        Mtm1 = Mt.copy()

        # Step 1: Update M
        Mt = np.dot(np.dot(M_tilde - Stm1, vone), vone.T) \
            + np.dot(np.dot(vone, vone.T), M_tilde - Stm1)

        # Step 2: Update S by soft thresholding
        St = _Slambda(M_tilde - Mt, lbda)

        if checks:
            np.testing.assert_allclose(St, -St.T, atol=1e-2, verbose=1)
            np.testing.assert_allclose(Mt, -Mt.T, atol=1e-2, verbose=1)

        # Step 3: convergence criterion
        # stop crit is convergence, not residual.
        converged = norm(Mt - Mtm1)/norm(Mt) < eps

    if verbose:
        print("Finished")
        print(norm(Mt - Mtm1)/norm(Mt))
        print(t)

    return Mt, St, norm(M_tilde - Mt - St) / norm(M_tilde)
