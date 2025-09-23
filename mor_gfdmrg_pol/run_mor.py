import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Any
import scipy.linalg
from evaltf import evaltf

# from mor_spectrum import mor_spectrum
from mor_spectrum2 import mor_spectrum2
from mor_spectrum_cs import mor_spectrum_all_c


# omega = np.linspace(-1.2, 0, 1201)
# eta = 0.005
# tol = 1e-3
real_shifts = False
make_plot = True


def run_mor(driver, dket, mpo, dmpo, ket, freqs: np.array, eta: float, tol: float,
            max_order:int=200, make_plot:bool=False, reload=False, mpi_comm=None, 
            savedir=None, **sweep_params):
    """
    :param driver: Block2 driver
    :param dket: excited state (dmpo @ ket)
    :param mpo: mpo with ground state energy subtracted from it
    :param dmpo: excitation mpo (at specific site)
    :param ket: ground state
    :param freqs: target frequencies to measure at
    :param eta: broadening parameter
    :param max_order: maximum iterations in MOR / maximum subspace size
    :param make_plot: make plot
    :param sweep_params: DMRG params for block2
    :return:
    """
    print('sweep params', sweep_params)
    sigma, sys, info = mor_spectrum2(driver, dket, mpo, dmpo, ket, freqs, eta, tol,
                                     max_order=max_order, make_plot=make_plot, 
                                     reload=reload, mpi_comm=mpi_comm,
                                     savedir=savedir, dmrg_opts=sweep_params)

    if make_plot:
        plt.figure()
        plt.plot(freqs, sigma)
        plt.xlabel('ω')
        plt.ylabel('absorption spectrum')
        plt.show()

    return sigma


def run_mor_all_c(driver, dket, mpo, dmpo, ket, freqs: np.array, eta: float, tol: float,
                  bra_exc_mpos: dict[str, Any],
                  max_order:int=200, make_plot:bool=False, reload=False, mpi_comm=None,
                  savedir=None, **sweep_params):
    """
    :param driver: Block2 driver
    :param dket: excited state (dmpo @ ket)
    :param mpo: mpo with ground state energy subtracted from it
    :param dmpo: excitation mpo (at specific site)
    :param ket: ground state
    :param freqs: target frequencies to measure at
    :param eta: broadening parameter
    :param max_order: maximum iterations in MOR / maximum subspace size
    :param make_plot: make plot
    :param sweep_params: DMRG params for block2
    :return:
    """
    print('sweep params', sweep_params)
    sigma_all, sys, info = mor_spectrum_all_c(driver, dket, mpo, dmpo, ket, freqs, eta, tol,
                                          bra_exc_mpos,
                                          max_order=max_order, make_plot=make_plot,
                                          reload=reload, mpi_comm=mpi_comm,
                                          savedir=savedir, dmrg_opts=sweep_params)

    if make_plot:
        plt.figure()
        for k, sigma in sigma_all.items():
            plt.plot(freqs, sigma, label=k)
        plt.xlabel('ω')
        plt.ylabel('absorption spectrum')
        plt.legend()
        plt.show()

    return sigma

