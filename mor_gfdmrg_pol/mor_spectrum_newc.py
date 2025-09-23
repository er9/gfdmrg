import pdb

import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import solve
from numpy.linalg import cond

from mor_sampling import mor_sampling
from mor_spectrum2 import mor_spectrum2, select_shift
from evaltf import evaltf, evaltf_new_c
from typing import Any





def load_new_c(driver, cstr, savedir=None) -> dict[str, Any]:

    fdir = driver.scratch if savedir is None else savedir
    print('fdir', fdir)
    At = np.load(fdir + 'At.npy')
    Et = np.load(fdir + 'Et.npy')
    bt = np.load(fdir + 'bt.npy')
    num_points = list(np.load(fdir + 'subspace_size.npy'))
    points = np.load(fdir + 'points.npy')
    assert (num_points[-1] == len(points)), f'should have {num_points} points, but have {len(points)} points selected'

    ct = np.load(fdir + f'{cstr}/' + 'ct.npy')
    print('loaded ct')
    conds = list(np.load(fdir + 'conds.npy'))
    sigmas = np.load(fdir + f'{cstr}/' + 'sigmas.npy')
    sigmas = [sigmas[i] for i in range(len(sigmas))]
    print('loaded sigmas')
    errs = np.load(fdir + f'{cstr}/' + 'errs.npy')
    errs = [errs[i] for i in range(len(errs))]

    Vs = []
    for i in range(num_points[-1]):
        try:
            orthog_vec = driver.load_mps(tag=f'ORTHOG_{i}')
            Vs += [orthog_vec]
        except FileNotFoundError:
            print(f"didn't find 'ORTHOG_{i}'")

    return {'At': At, 'Et': Et, 'bt': bt, 'ct': ct, 'Vs': Vs, 'points': points,
            'conds': conds, 'sigmas': sigmas, 'errs': errs, 'subspace_size': num_points}


def save_new_c(driver, c_str: str, ct: np.ndarray, points: list,
               conds: list, sigmas: list, errs: list, subspace_size: list, savedir=None):

    fdir = (driver.scratch if savedir is None else savedir) + '/' + c_str + '/'
    np.save(fdir + 'ct', ct)
    np.save(fdir + 'conds', conds)
    np.save(fdir + 'sigmas', sigmas)
    np.save(fdir + 'errs', errs)
    np.save(fdir + 'subspace_size', subspace_size)
    np.save(fdir + 'points', points)
    print('SAVED FILES FOR NEW C')
    return

def mor_spectrum_new_c(driver, gf_ket, mpo, exc_mpo, gs_ket, freqs, eta, tol, bra_exc_mpo,
                       max_order=200, make_plot=False, dmrg_opts=None, reload=True, orthogonalize=False, savedir=None):

    try:
        print('reload?', reload)
        if not reload:
            raise FileNotFoundError

        print('load data', savedir)
        data = load_new_c(driver, bra_exc_mpo.tag, savedir=savedir)
        Sigmas = data['sigmas']
        Errs = data['errs']
        Conds = data['conds']
        O = data['subspace_size']
        if len(Errs) == 0:
            e = np.inf * np.ones_like(freqs)
        else:
            e = Errs[-1]  ## Sigmas[-1] - Sigmas[-2]
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

        return mor_spectrum2(driver, gf_ket, mpo, exc_mpo, gs_ket, freqs, eta, tol, bra_exc_mpo=bra_exc_mpo,
                             max_order=max_order, make_plot=make_plot, dmrg_opts=dmrg_opts, reload=reload,
                             orthogonalize=orthogonalize)

    prev_ct = None
    Sigmas = []
    Errs = []
    for it, num_sample in enumerate(range(2, len(freqs_sample))):
        y, ct = evaltf_new_c(driver, At, Et, bt, bra_exc_mpo, gs_ket, freqs + 1j *eta, V, num_sample,
                             prev_ct=prev_ct)
        sigma = -1 / np.pi * np.imag(y)
        prev_ct = ct

        Sigmas.append(sigma)
        if it > 0:
            e = np.abs(sigma / np.max(sigma) - Sigmas[-2] / np.max(Sigmas[-2]))
            Errs.append(e)

    c_str = bra_exc_mpo.tag
    save_new_c(driver, c_str, ct, freqs_sample, Sigmas, Errs, savedir=savedir)

    # outputs
    sys = {'A': At, 'E': Et, 'b': bt, 'c': ct, 'V': V}
    info = {'shifts': freqs_sample, 'Sigma_rel': np.array(Sigmas), 'Err': np.array(Errs), 'Cond': np.array(Conds)}

    return sigma, sys, info






