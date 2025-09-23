import pdb

import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import solve
from numpy.linalg import cond

from mor_sampling import mor_sampling
from evaltf import evaltf
from typing import Any


def select_shift(shifts, omega, err, tol):
    N = len(omega)
    check = err <= tol
    if np.sum(check) == N:
        shift = None
    else:
        idx = np.argmax(err)
        w = np.real(omega[idx])
        rs = np.sort(np.real(shifts))  # real shifts (sorted)
        cs = shifts[np.argsort(np.real(shifts))]  # complex shifts (sorted)
        print('max err w', w, 'err', err[idx])
        print('rs', rs)
        print('cs', cs)
        print(rs < w)
        print(rs > w)
        idx0 = np.argmin(np.array(rs < w))
        idx1 = np.argmax(np.array(rs > w))
        print(idx0, idx1)
        if idx0 == idx1:
            idx = idx0 - 1
        elif idx1 == idx0 + 1:
            idx = idx0
        else:
            raise ValueError # w chosen outside of range
        print('idx', idx)
        # print(np.where((rs[:-1] < w) & (w < rs[1:])))
        # idx = np.max(np.where((rs[:-1] < w) & (w < rs[1:])))
        # print('idx', idx)
        shift = (cs[idx] + cs[idx+1]) / 2
        shifts = np.append(shifts, shift)
    return shift, shifts


def update_plot(omega, sigma, shifts, e, tol, O, i):
    # check inputs and defaults
    if isinstance(shifts, list):
        sel = np.array(shifts).flatten()
    else:
        sel = shifts

    # spectrum
    if i % 2 == 0:
        plt.subplot(3, 1, 1)
    else:
        plt.subplot(3, 1, 2)
    plt.cla()
    plt.plot(omega, sigma, 'k')
    plt.plot(np.real(sel), np.zeros_like(sel), '+')
    plt.ylabel('spectrum')
    plt.legend([f'k = {len(sel)}'])
    if tol is not None and i % 2 == 0:
        plt.title(f'tol = {tol}')

    # relative error
    plt.subplot(3, 1, 3)
    plt.cla()
    if e is not None:
        plt.semilogy([omega[0], omega[-1]], [tol, tol], ':k')
        plt.semilogy(omega, e)
        l = f'|σ_{O[i+1]} - σ_{O[i]}|'
        plt.legend(['', l])
    plt.xlabel('ω')
    plt.ylabel('relative error')
    plt.ylim([1e-10, 1e0])
    plt.pause(1)



def compute_gf(driver, dket, mpo, dmpo, ket, freq, eta, **sweep_params):
    bra = driver.copy_mps(dket, tag="BRA")  # initial guess
    gf_out = driver.greens_function(bra, mpo, dmpo, ket, freq, eta,
                                    **sweep_params)
                                    # n_sweeps=6, bra_bond_dims=[200], ket_bond_dims=[200],
                                    # noises=[1e-5] * 4 + [0],
                                    # thrds=[1E-6] * 10, iprint=2)
    print("FREQ = %8.2f GF = %12.6f + %12.6f i" % (freq, gf_out.real, gf_out.imag))

    return gf_out, bra


def load(driver, savedir=None) -> dict[str, Any]:

    fdir = driver.scratch + '/' if savedir is None else savedir
    print('fdir', fdir)
    At = np.load(fdir + 'At.npy')
    Et = np.load(fdir + 'Et.npy')
    bt = np.load(fdir + 'bt.npy')
    ct = np.load(fdir + 'ct.npy')
    print('loaded ct')
    conds = list(np.load(fdir + 'conds.npy'))
    sigmas = np.load(fdir + 'sigmas.npy')
    sigmas = [sigmas[i] for i in range(len(sigmas))]
    print('loaded sigmas')
    errs = np.load(fdir + 'errs.npy')
    errs = [errs[i] for i in range(len(errs))]
    num_points = list(np.load(fdir + 'subspace_size.npy'))
    points = np.load(fdir + 'points.npy')
    assert(num_points[-1] == len(points)), f'should have {num_points} points, but have {len(points)} points selected'

    Vs = []
    for i in range(num_points[-1]):
        try:
            orthog_vec = driver.load_mps(tag=f'ORTHOG_{i}')
            Vs += [orthog_vec]
        except FileNotFoundError:
            print(f"didn't find 'ORTHOG_{i}'")

    return {'At': At, 'Et': Et, 'bt': bt, 'ct': ct, 'Vs': Vs, 'points': points,
            'conds': conds, 'sigmas': sigmas, 'errs': errs, 'subspace_size': num_points}


def save(driver, At: np.ndarray, Et: np.ndarray, bt: np.ndarray, ct: np.ndarray, points: list,
         conds: list, sigmas: list, errs: list, subspace_size: list,):

    fdir = driver.scratch + '/'
    if driver.mpi.rank == 0:
        np.save(fdir + 'At', At)
        np.save(fdir + 'Et', Et)
        np.save(fdir + 'bt', bt)
        np.save(fdir + 'ct', ct)
        np.save(fdir + 'conds', conds)
        np.save(fdir + 'sigmas', sigmas)
        np.save(fdir + 'errs', errs)
        np.save(fdir + 'subspace_size', subspace_size)
        np.save(fdir + 'points', points)
        print('SAVED FILES', fdir)
    return

def mor_spectrum2(driver, dket, mpo, dmpo, ket, freqs, eta, tol, bra_exc_mpo=None, max_order=200,
                  make_plot=False, dmrg_opts=None, savedir=None,
                  reload=True, orthogonalize=True, mpi_comm=None):
    """
    dket: V|gs> (used as initial guess for bra when solving Green's function)
    mpo: Hamiltonian
    dmpo: excitation Hamiltonian (used in Green's fct calculation, correction vector)
    ket: ket to solve for
    freqs: frequencies of interest
    eta: broadening factor
    tol: convergence tolerance
    """

    try:
        print('reload?', reload)
        if not reload:
            raise FileNotFoundError

        print('load data', savedir)
        data = load(driver, savedir=savedir)
        print('load data', savedir)
        Sigmas = data['sigmas']
        print('load sigmas')
        Errs = data['errs']
        Conds = data['conds']
        O = data['subspace_size']
        if len(Errs) == 0:
            e = np.inf * np.ones_like(freqs)
        else:
            e = Errs[-1]    ## Sigmas[-1] - Sigmas[-2]
        current_order = len(Sigmas)

        freqs_sample = data['points']
        At = data['At']
        Et = data['Et']
        bt = data['bt']
        ct = data['ct']
        V = data['Vs']

        print('size', len(Sigmas), len(Errs), len(Conds), len(O), current_order)
        print('At', At.shape, Et.shape, bt.shape, ct.shape, len(V))
        print('freqs sample', freqs_sample)

    except FileNotFoundError:

        Sigmas = []
        Errs = []
        Conds = []
        O = [0]
        e = np.inf * np.ones_like(freqs)
        current_order = 0

        V = None
        At, Et, bt, ct = np.empty((0,0)), np.empty((0,0)), np.empty((0,)), np.empty((0,))

    for it in range(current_order, max_order):
        if it == 0:
            # freqs_sample = np.array([freqs[0], freqs[-1]])
            freqs_sample = np.array([freqs[-1], freqs[0]])
            # freqs_sample = np.linspace(freqs[0], freqs[-1], 16, endpoint=True)
            # freqs_sample = np.arange(freqs[0], freqs[-1], 0.02)
            print('it0 freqs sample', freqs_sample)
            selection = freqs_sample
        elif it == 1:
            selection = np.array([freqs[len(freqs)//2]])
            freqs_sample = np.append(freqs_sample, selection)
        else:
            new_sample, freqs_sample = select_shift(freqs_sample, freqs, e, tol)
            if new_sample is None:
                break
            selection = np.array([new_sample])

        print(f'* Level {it + 1}: ')
        # update reduced system
        print('selection', selection)
        orthogonalize=False
        print('orthogonalize', orthogonalize)
        if dmrg_opts is None:
            dmrg_opts = {}
        # dmrg_opts.setdefault('tol', tol)
        A_dict, E_dict, b_dict, c_dict, V, = mor_sampling(driver, dket, mpo, dmpo, ket, selection, eta,
                                                          bra_exc_mpo=bra_exc_mpo, V=V,
                                                          orthogonalize=orthogonalize, dmrg_opts=dmrg_opts, 
                                                          mpi_comm=mpi_comm)

        #### fill in At, Et, bt, ct matrices ###
        m = len(V)

        At_new = np.empty((m, m), dtype=complex)
        ## previous entries
        m_, n_ = At.shape
        At_new[:m_, :n_] = At
        ## new entries
        for i, j in A_dict.keys():
            print('i,j', i, j)
            At_new[i, j] = A_dict[(i, j)]
            if i != j:
                At_new[j, i] = np.conj(At_new[i, j])  ## assumes Hermitian operator
        At = At_new

        Et_new = np.empty((m, m), dtype=complex)
        m_, n_ = Et.shape
        Et_new[:m_, :n_] = Et
        for i,j in E_dict.keys():
            Et_new[i,j] = E_dict[(i,j)]
            if i != j:
                Et_new[j, i] = np.conj(Et_new[i, j])    ## assumes Hermitian operator
        Et = Et_new

        bt_new = np.empty((m,), dtype=complex)
        m_, = bt.shape
        bt_new[:m_] = bt
        for i, in b_dict.keys():
            print('i', i)
            bt_new[i] = b_dict[(i,)]
        bt = bt_new

        ct_new = np.empty((m,), dtype=complex)
        m_, = ct.shape
        ct_new[:m_] = ct
        for i, in c_dict.keys():
            print('i', i)
            ct_new[i] = c_dict[(i,)]
        ct = ct_new

        condEt = cond(Et)
        Conds.append(condEt)
        print(f'  cond(Et) = {condEt:.2e}: ')

        # evaluate reduced system
        sigma = -1 / np.pi * np.imag(evaltf(At, Et, bt, ct, freqs + 1.j*eta))
        sigman = sigma / np.max(sigma)  # normalization
        Sigmas.append(sigma)
        O.append(len(V))

        # relative error
        if it > 0:
            e = np.abs(sigman - Sigmas[-2]/np.max(Sigmas[-2]))
            Errs.append(e)

        save(driver, At, Et, bt, ct, freqs_sample, Conds, Sigmas, Errs, O)

        # plot absorption spectrum
        if make_plot:
            update_plot(freqs, sigma, freqs_sample, e, tol, O, it)

    # outputs
    sys = {'A': At, 'E': Et, 'b': bt, 'c': ct, 'V': V}
    info = {'shifts': freqs_sample, 'Sigma_rel': np.array(Sigmas), 'Err': np.array(Errs), 'Cond': np.array(Conds)}

    return -1 / np.pi * np.imag(evaltf(At, Et, bt, ct, freqs + 1j*eta)), sys, info




