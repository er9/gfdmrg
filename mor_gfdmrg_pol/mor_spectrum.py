from typing import Union
import numpy as np
from scipy.sparse.linalg import gmres
from scipy.linalg import qr
import matplotlib.pyplot as plt

from mor_sampling import mor_sampling
from evaltf import evaltf

def mor_spectrum(A: np.ndarray, E: np.ndarray, b: np.ndarray, c: np.ndarray, omega: np.ndarray, eta: float,
                 k: Union[int, float], use_real_shifts:bool=False, make_plot:bool=False):
    """
    Absorption spectrum via model order reduction.

    Parameters:
    :param A: (numpy array) System matrix A.
    :param E: (numpy array) System matrix E.
    :param b: (numpy array) Input vector b.
    :param c: (numpy array) Output vector c.
    :param omega: (numpy array) Frequencies at which to evaluate the absorption spectrum.
    :param eta: (float) Broadening factor.
    :param k: (int or float) Order of the reduced order model or tolerance.
    :param use_real_shifts: (bool, optional) Use real interpolation points. Defaults to False.
    :param make_plot: (bool, optional) Generate plots for adaptive order determination. Defaults to False.

    :return:
    sigma: (numpy array) Absorption spectrum at frequencies omega.
    sys: (dict) Reduced order model.
    info: (dict) Information about the adaptively selected shifts.
    """
    n = A.shape[0]

    shifts = generate_shifts(omega, eta, k, use_real_shifts)
    omega_cmplx = omega + 1j*eta

    if isinstance(k, int):
        # Reduced order model of dimension k
        At, Et, bt, ct, V, info = mor_sampling(A, E, b, c, shifts)
        sigma = -1 / np.pi * np.imag(evaltf(At, Et, bt, ct, omega_cmplx))

        fig1, ax1 = plt.subplots()

        if make_plot:
            ax1.plot(omega, sigma, 'k')
            ax1.set_xlabel('ω')
            ax1.set_ylabel('absorption spectrum')
            ax1.legend([f'k = {len(shifts)}'])

        sys = {'A': At, 'E': Et, 'b': bt, 'c': ct, 'V': V}
        info = {'shifts': shifts}

    else:
        if make_plot:
            fig1, ax1s = plt.subplots(2, 1)
            ax1, ax2 = ax1s

        # Adaptive shift selection loop
        V = np.zeros((n, 0))                ## selected subspace
        Sigma = np.zeros((len(omega), 0))   ## DOS at desired freqs
        Err = np.zeros((len(omega), 0))     ## Error at desired freqs
        O = np.zeros(1, dtype=int)          ## number of vectors in subspace at each level
        e = np.inf * np.ones(len(omega_cmplx))    ## errors at desired freqs
        sel = []

        ## for each new shift point
        for i in range(len(shifts)):
            print('shifts i', i)

            ## select shifts in unconverged intervals
            selection = select_shifts(shifts[i], omega_cmplx, e, k)     ## k: tolerance
            print('selection points', selection)
            if not selection.size:  ## no selected points
                break
            else:
                print(f'* Level {i} ({len(selection)} shifts)')
                sel += [selection]

            ## update reduced system
            At, Et, bt, ct, V, info = mor_sampling(A, E, b, c, selection, V)

            ## evaluate reudced system
            sigma = -1 / np.pi * np.imag(evaltf(At, Et, bt, ct, omega_cmplx))
            sigman = sigma / np.max(sigma)
            Sigma = np.hstack((Sigma, sigman[:, None]))
            print('Sigma shape', Sigma.shape)
            O = np.append(O, V.shape[1])
            print('O', O)

            ## relative error
            if i > 0:
                e = np.abs(sigman - Sigma[:, i-1])
                Err = np.hstack((Err, e[:, None]))
                print('Err shape', Err.shape)

            ## plot spectrum
            if make_plot:
                update_plot(ax1, ax2, omega, sigma, selection, e, k, O, i)

        sys = {'A': At, 'E': Et, 'b': bt, 'c': ct, 'V': V}
        info = {'shifts': sel, 'Sigma_rel': Sigma, 'Err': Err}

    return sigma, sys, info


def generate_shifts(omega_, eta_, k_, use_real_shifts_:bool):
    """
    :param omega_: interpolation points
    :param eta_: broadening factor
    :param k_: k is dimension of reduced order model
    :param use_real_shifts_: if shifts are real (eta_ = 0)
    :return: new interpolation points?
    """
    omin = omega_[0]
    omax = omega_[-1]

    if isinstance(k_, int):
        shifts = np.linspace(omin, omax, k_)
        if not use_real_shifts_:
            shifts = shifts + 1j * eta_
    else:
        # k is tolerance for reduced order model
        npts = 3
        levels = 16

        shifts = [np.linspace(omin, omax, npts)]
        for i in range(1, levels):
            npts = 2 * npts - 1
            t = np.linspace(omin, omax, npts)
            shifts.append(t[1::2])

        if not use_real_shifts_:
            shifts = [s + 1j * eta_ for s in shifts]

    return shifts

def select_shifts(candidates_, omega_, fun, tol):
    """
    :param candidates_: candidate points
    :param omega_: frequencies to evaluate at
    :param fun: error
    :param tol: cutoff tolerance
    :return: new interpolation points
    """
    K = len(candidates_)
    N = len(omega_)

    check = fun <= tol
    shifts_ = np.array([])

    k = 0
    b = candidates_[k].real + (candidates_[k + 1].real - candidates_[k].real) / 2
    idx = 0

    for i in range(N):
        if omega_[i].real >= b:
            # check previous interval
            # print('idx', idx, i)
            # print('check', fun[idx:i])
            if not np.all(check[idx:i]):
                shifts_ = np.append(shifts_, candidates_[k])
            idx = i
            # update interval
            k += 1
            if k < K - 1:
                b = candidates_[k].real + (candidates_[k + 1].real - candidates_[k].real) / 2
            else:
                b = omega_[-1].real

    return shifts_

def update_plot(ax1_, ax2_, omega_, sigma_, shifts_, e_, tol, O_, i_):
    """
    :param ax1_: mpl Axis object (plot ldos)
    :param ax2_: mpl Axis object (plot error)
    :param omega_: desired evaluated frequencies
    :param sigma_: spectrum values
    :param shifts_: measured points
    :param e_: error
    :param tol: tolerance
    :param O_: ?
    :param i_: ?
    :return: None
    """
    print('O', O_.shape)

    # check inputs and defaults
    if isinstance(shifts_, list):
        sel = np.array(shifts_).flatten()
    else:
        sel = shifts_

    # absorption spectrum
    ax1_.plot(omega_, sigma_, 'k')
    ax1_.plot(sel.real, np.zeros_like(sel), '+')
    ax1_.set_xlabel('ω')
    ax1_.set_ylabel('absorption spectrum')
    ax1_.legend([f'k = {len(sel)}'])
    if tol is not None:
        ax1_.set_title(f'tol = {tol}')

    # relative error
    if e_ is not None:
        if O_[i_] == 0:
            ax2_.semilogy([omega_[0], omega_[-1]], [tol, tol], ':k')
        else:
            ax2_.semilogy(omega_, e_)
            k0 = O_[i_]
            k1 = O_[i_ + 1]
            l = f'|σ_{k1} - σ_{k0}|'
            if i_ < 3:
                ax2_.legend(['', l])
            else:
                ax2_.legend([''] + [f'σ_{j}' for j in range(1, i_ + 1)] + [l])

    ax2_.set_xlabel('ω')
    ax2_.set_ylabel('relative error')
    ax2_.set_ylim([1e-15, 1e0])

    return
