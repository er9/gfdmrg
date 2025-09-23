import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import gto, scf, lo

from run_mor import run_mor

""" shows how to properly treat different kinds of symmetry
"""

BOHR = 0.52917721092
R = 1.8 * BOHR
N = 6

symm = [
    SymmetryTypes.SU2,
    SymmetryTypes.SU2 | SymmetryTypes.CPX,
    SymmetryTypes.SZ,
    SymmetryTypes.SZ | SymmetryTypes.CPX,
    SymmetryTypes.SGF,
    SymmetryTypes.SGFCPX
][2]

mol = gto.M(atom=[['H', (i * R, 0, 0)] for i in range(N)], basis="sto6g", symmetry="c1", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1E-14)

mf.mo_coeff = lo.orth.lowdin(mol.intor('cint1e_ovlp_sph'))
if SymmetryTypes.SU2 in symm or SymmetryTypes.SZ in symm:
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, ncore=0, ncas=None, g2e_symm=8)
else:
    gmf = mf.to_ghf()
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_ghf_integrals(gmf, ncore=0, ncas=None)

print('ncas', ncas, 'nelec', n_elec)

driver = DMRGDriver(scratch="./tmp", symm_type=symm, n_threads=4)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

bond_dims = [150] * 4 + [200] * 4
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8
tol = 1.0e-3

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, integral_cutoff=1E-8, iprint=1)
ket = driver.get_random_mps(tag="KET", bond_dim=150)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
    thrds=thrds, iprint=1)
print('ket.nsites', ket.n_sites)
print('Ground state energy = %20.15f' % energy)

# if SymmetryTypes.SU2 in symm or SymmetryTypes.SZ in symm:
#     isite = 2
# else:
#     isite = 2 * 2 # SGF uses spin orbitals


mpo.const_e -= energy
tot_ldos = 0.0
for ex_site in [2]:  # range(ket.n_sites):

    # if SymmetryTypes.SU2 in symm or SymmetryTypes.SZ in symm:
    #     isite = ex_site
    # else:
    #     isite = ex_site * 2 # SGF uses spin orbitals

    isite = ex_site
    eta = 0.005

    # dmpo = driver.get_identity_mpo()
    dmpo = driver.get_site_mpo(op='D', site_index=isite, iprint=0)
    if SymmetryTypes.SU2 in symm:
        # SU2 mode needs special treatment of non-singlet states
        dket = driver.get_random_mps(tag="DKET", bond_dim=200, center=ket.center, left_vacuum=dmpo.left_vacuum)
    else:
        dket = driver.get_random_mps(tag="DKET", bond_dim=200, center=ket.center,
                                     target=dmpo.op.q_label + ket.info.target)

    driver.multiply(dket, dmpo, ket, n_sweeps=10, bond_dims=[200], thrds=[1E-10] * 10, iprint=0)

    ## MOR sampling
    freqs = np.arange(-1.2, 0, 0.01)
    ldos = run_mor(driver, dket, mpo, dmpo, ket, freqs, eta, tol, make_plot=True,
                    n_sweeps=6, bra_bond_dims=[200], ket_bond_dims=[200],
                    noises=[1e-5] * 4 + [0],
                    thrds=[1E-6] * 10, # iprint=2
                    reload=False
                    )
    # print("FREQ = %8.2f GF[%d,%d] = %12.6f + %12.6f i" % (freq, isite, isite, gfmat[iw].real, gfmat[iw].imag))
    tot_ldos += ldos

import matplotlib.pyplot as plt
plt.grid(which='major', axis='both', alpha=0.5)
plt.plot(freqs, tot_ldos, linestyle='-', marker='o', markersize=4, mfc='white', mec="#7FB685", color="#7FB685")
plt.xlabel("Frequency $\\omega$")
plt.ylabel("LDOS")
plt.show()


#
# print('?')
# freqs = np.arange(-1.0, -0.2, 0.01)
# gfmat = np.zeros((len(freqs), ), dtype=complex)
# for iw, freq in enumerate(freqs):
#     bra = driver.copy_mps(dket, tag="BRA") # initial guess
#     gfmat[iw] = driver.greens_function(bra, mpo, dmpo, ket, freq, eta, n_sweeps=6,
#         bra_bond_dims=[200], ket_bond_dims=[200], thrds=[1E-6] * 10, iprint=0)
#     print("FREQ = %8.2f GF[%d,%d] = %12.6f + %12.6f i" % (freq, isite, isite, gfmat[iw].real, gfmat[iw].imag))
#
# ldos = -1 / np.pi * gfmat.imag
#
# if SymmetryTypes.SU2 not in symm:
#     # SU2 mode computes the sum of alpha and beta spins
#     # in SZ and SGF mode only one spin is computed
#     ldos *= 2
#
#
# symm_str = symm.__repr__()
# symm_str = symm_str.split('.')[1]
# symm_str = symm_str.split(':')[0]
#
# import matplotlib.pyplot as plt
# plt.grid(which='major', axis='both', alpha=0.5)
# plt.plot(freqs, ldos, linestyle='-', marker='o', markersize=4, mfc='white', mec="#7FB685", color="#7FB685")
# plt.xlabel("Frequency $\\omega$")
# plt.ylabel("LDOS")
# # plt.savefig('test-%s.png' % symm_str)
# plt.show()