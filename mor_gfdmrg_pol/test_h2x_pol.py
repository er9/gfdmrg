import sys, getopt, pickle, os
import shutil
import numpy as np

# from matplotlib import pyplot as plt
from pyscf import scf, mcscf, gto, lib
from pyblock2._pyscf import scf as b2scf
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SOCDMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from matplotlib import pyplot as plt
from run_mor import run_mor_all_c

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# use_mpi =True

use_mpi = False
comm = None
rank = 0
size = 1

"""
polarization propagator
https://kthpanor.github.io/echem/docs/spec_prop/adc_pp.html
Eq.3.60 of https://publikationen.ub.uni-frankfurt.de/opus4/frontdoor/deliver/index/docId/7038/file/Dissertation.pdf
"""

ha2ev = -27
peak_loc_ev = {'H2O': 408.22,
               'H2S': 2498.76  ,    ## observed at ~ -91.7 ha
               'H2Se': 12726.01,
               'H2Te': 31959.75,
               'H2Po': 93589.78,
              }

shift_ev = {'H2O': 0.0,
            'H2S': -23.5,
            'H2Se': -102.,
            'H2Te': -252.,
            'H2Po': -730.,
            }

# ## medium scale
# fmin_ev = {'H2O': 0.0,
#            'H2S': -30,
#            'H2Se': -110.,
#            'H2Te': -260.,
#            'H2Po': -740.,
#            }
#
# fmax_ev = {'H2O': 0.0,
#            'H2S': -20.,  # 10
#            'H2Se': -95.,  # -70.,
#            'H2Te': -240.,  # -220.,
#            'H2Po': -720.,  # -680.,
#            }
#
# ### fine scale
# fmin_ev = {'H2O': 0.0,
#            'H2S': -24,
#            'H2Se': -103.5,  # -110.,
#            'H2Te': -252.5,  # -260.,
#            'H2Po': -731.,  # -740.,
#            }
#
# fmax_ev = {'H2O': 0.0,
#            'H2S': -22.,
#            'H2Se': -102.,  # -95.,  # -70.,
#            'H2Te': -250.,  # -240.,  # -220.,
#            'H2Po': -728.,  # -720.,  # -680.,
#            }

#### parameters that can be set via flags
restart = False
load_gs = False
eta = 0.02  # 0.005  # 0.02
bond_dim = 300
scratch_folder = 'ip'
# fmin, fmax, df = -22.0, -18.0, 0.1
# fmin, fmax, df = -20.0, -19.25, 0.01
fmin, fmax, df = None, None, None
# fmin, fmax, df = -1, 1, 0.1
tol = 1e-5  # 0.001

molec = 'H2S'
p, q = 0, 0

def main(): 

    scratch_folder = f'mor_ip_no_{molec}_mpi1/eta{eta}_b{bond_dim}_v2/'
    # fcidump_fstr = f'/global/homes/e/erikaye/gfdmrg/gfdmrg/ryanChalk/K_xanes/fcidump_{molec}/FCIDUMP'
    # fcidump_fstr = f'/global/homes/e/erikaye/gfdmrg/gfdmrg/ryanChalk/K_xanes/fcidump_{molec}/FCIDUMP'
    fcidump_fstr = f'../gfdmrg_h2x/fcidump_{molec}/FCIDUMP'

    print('fs', fmin, fmax, df)

    exc_str = f'c{p}d{q}'

    root_dir = './'  # /pscratch/sd/e/erikaye/gfdmrg/
    freq_str = f'fs_{fmin}_{fmax}_{df}'
    SCRATCH_GS = f"./gse_{molec}/"
    SCRATCHDIR = f"{root_dir}/{scratch_folder}/{freq_str}/{exc_str}/"
    SCRATCHDIR_LDOS =  f"{root_dir}/{scratch_folder}/{freq_str}/ldos/"
    # SCRATCHDIR = f"/pscratch/sd/e/erikaye/gfdmrg/{scratch_folder}{exc_site}/"
    # SCRATCHDIR_LDOS = SCRATCHDIR + f"ldos/" + freq_str + '/'
    if not os.path.exists(SCRATCHDIR):
        os.makedirs(SCRATCHDIR, exist_ok=True)
    if not os.path.exists(SCRATCHDIR_LDOS):
        os.makedirs(SCRATCHDIR_LDOS, exist_ok=True)
    print('scratch dir', SCRATCHDIR)
    print('scratch dir ldos', SCRATCHDIR_LDOS)

    """
    fyi: https://block2.readthedocs.io/en/latest/tutorial/dmrg-soc.html
    start with 1 extra electron
    have the option of doing multi-root DMRG or computing ionization DOS via DDMRG
    """
    
    nroots = 1              # if doing multi-root DMRG, use this many roots
    symm = SymmetryTypes.SGFCPX
    
    driver = SOCDMRGDriver(scratch=SCRATCHDIR, # n_threads=64, 
                           symm_type=symm, mpi=use_mpi)
    # driver = SOCDMRGDriver(scratch=SCRATCHDIR, symm_type=symm, mpi=True)
    print('fcidump', fcidump_fstr)
    driver.read_fcidump(filename=fcidump_fstr, iprint=2) # , pg='d2h')
    print('loaded info:', '\nnum sites', driver.n_sites, '\nnum elec', driver.n_elec,
          '\nspin', driver.spin, '\norb symmetry', driver.orb_sym)
    
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec + 1,
                             spin=driver.spin + 1, orb_sym=driver.orb_sym)
    ## note: this does not update the n_elec, spin of the driver

    mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1)  ## will this work before "initialize_system"?
    impo = driver.get_identity_mpo()
    n_sites = driver.n_sites

    if driver.mpi is not None:
        driver.mpi.barrier()


    ## compute groud state with one extra electron
    if not load_gs: 
        # bond_dims = [150] * 4 + [200] * 4
        # bond_dims = [250] * 4 + [300] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        
        gs_ket_ = driver.get_random_mps(tag=f"GS_KET", bond_dim=bond_dim)
        gs_energy = driver.dmrg(mpo, gs_ket_, n_sweeps=20, # bond_dims=bond_dims, noises=noises,
                                thrds=thrds, iprint=1)

        print('Ground state energy = %20.15f' % gs_energy)

        ## save to SCRATCH_GS
        os.makedirs(SCRATCH_GS, exist_ok=True)
        gs_tag = "GS_KET"
        if SCRATCH_GS != driver.scratch:
            for k in os.listdir(driver.scratch):
                if f'.{gs_tag}.' in k or k == f'{gs_tag}-mps_info.bin':
                    shutil.copy(driver.scratch + "/" + k, SCRATCH_GS + "/" + k)

        if driver.mpi is not None:
            driver.mpi.barrier()
        gs_ket = driver.copy_mps(gs_ket_, f"GS_KET_{exc_str}")
        if driver.mpi is not None:
            driver.mpi.barrier()


        # print('check copy energy')
        # gs_energy = driver.expectation(gs_ket, mpo, gs_ket)
        # norm = driver.expectation(gs_ket, impo, gs_ket)
        # print('gs energy', gs_energy, norm)
        # print('Ground state energy = %20.15f' % (gs_energy/norm))

        # print('check orig energy')
        # gs_energy = driver.expectation(gs_ket_, mpo, gs_ket_)
        # norm = driver.expectation(gs_ket_, impo, gs_ket_)
        # print('gs energy', gs_energy, norm)
        # print('Ground state energy = %20.15f' % (gs_energy/norm))
    else:
        if driver.mpi is not None:
            driver.mpi.barrier()
        gs_tag = "GS_KET"
        try:
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)
        except:
            restart_dir = SCRATCH_GS
            if restart_dir != driver.scratch:
                for k in os.listdir(restart_dir):
                    if f'.{gs_tag}.' in k or k == f'{gs_tag}-mps_info.bin':
                        shutil.copy(restart_dir + "/" + k, driver.scratch + "/" + k)
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)

        gs_ket = driver.copy_mps(gs_ket, f"GS_KET_{exc_str}")
        if driver.mpi is not None:
            driver.mpi.barrier()
        norm = np.abs(driver.expectation(gs_ket, impo, gs_ket)) / (1 if driver.mpi is None else driver.mpi.size)
        gs_energy = driver.expectation(gs_ket, mpo, gs_ket)
        print('gs energy', gs_energy, norm)
        print(f'Ground state energy = {gs_energy/norm}')
    

    ###################
    ## D-DMRG
    ###################
    print('start dynamical DMRG')

    ## update scratch to avoid overwriting?  ## messes up gs_ket calling then...
    # driver.scratch = SCRATCHDIR_LDOS

    mpo.const_e -= gs_energy
    # eta = 0.1  ## TlH   (examples tend to use 0.0005)
    print('eta', eta)
    freqs = np.arange(fmin, fmax, df)
    print('len freqs', len(freqs))
    
    tot_ldos = np.zeros((len(freqs),), dtype=complex)
    for it in [0]:   ## only do + excitation (ignore - excitation)

        print(f'excite p{p}, q{q}', gs_ket.n_sites)
    
        if p > n_sites or q > n_sites:
            raise ValueError

        try:
            raise IOError
            ldos = pickle.load(open(SCRATCHDIR_LDOS + f'a{isite}' + '.pkl','rb'))
            print('loaded ' + SCRATCHDIR_LDOS + f'a{isite}' )
            # print('ldos', ldos)

        except:

            # ## electron excitation at site # isite (annihilation)
            # dmpo = driver.get_site_mpo(op='D', site_index=isite, iprint=0)  
            
            ## c: creation, d: annihilation
            b = driver.expr_builder()
            print('p,q', p, q)
            b.add_term("CD", [p, q], 1.0)   ## does this work if i=j?
            exc_mpo = driver.get_mpo(b.finalize(), iprint=0)

            ## c: creation, d: annihilation
            bra_exc_mpos = {}
            for r in range(2):
                for s in range(2):
                    b = driver.expr_builder()
                    b.add_term("CD", [r, s], 1.0)  ## does this work if i=j?
                    bra_exc_mpo = driver.get_mpo(b.finalize(), iprint=0)
                    exc_str = f'c{r}d{s}'
                    bra_exc_mpos[exc_str] = bra_exc_mpo

            ## initial guess for GF calculation
            dket = driver.get_random_mps(tag=f"DKET{exc_str}", bond_dim=bond_dim,
                                         center=gs_ket.center,
                                         target=exc_mpo.op.q_label + gs_ket.info.target)
            print('d multiply')
            driver.multiply(dket, exc_mpo, gs_ket, n_sweeps=10, # bond_dims=[bond_dim],
                            # thrds=[1E-10] * 10,
                            iprint=0)

            ### dket = dmpo * ket  (ket with removed electron)
            print('done d multiply', bond_dim)
            print('start mor')
            sys.stdout.flush()

            ### MOR sampling

            ldos = run_mor_all_c(driver, dket, mpo, exc_mpo, gs_ket, freqs if it == 0 else -freqs, eta, tol,
                                 bra_exc_mpos=bra_exc_mpos,
                                 max_order=10, make_plot=True,
                                 n_sweeps=20, bra_bond_dims=[bond_dim], ket_bond_dims=[bond_dim],
                                 # noises=[1e-5] * 4 + [0],
                                 # tol=1.0e-3,
                                 # thrds=[1.0e-3] * 10,  # iprint=
                                 reload=True,
                                 mpi_comm=comm,
                                 )
            # print("FREQ = %8.2f GF[%d,%d] = %12.6f + %12.6f i" % (freq, isite, isite, gfmat[iw].real, gfmat[iw].imag))
            
            if driver.mpi is None or driver.mpi.rank == 0:
                pickle.dump(ldos, open(SCRATCHDIR_LDOS + f'a{isite}' + '.pkl','wb'))

        if driver.mpi is None or driver.mpi.rank == 0:
            plt.figure()
            plt.grid(which='major', axis='both', alpha=0.5)
            freqs = np.arange(fmin, fmax, df)
            plt.semilogy(freqs * -27.2, ldos, linestyle='-', marker='o', markersize=4,
                         mfc='white', mec="#7FB685", color="#7FB685")
            plt.xlabel("Frequency $\\omega$ [eV]")
            plt.ylabel("LDOS")
            plt.savefig(f'ldos_freqs_{freq_str}_eta{eta}_b{bond_dim}_{exc_str}.png')
            plt.show()
            plt.close()



#################################
## set parameters via input flags
#################################


opts, args = getopt.getopt(sys.argv[1:], "rla:b:f:F:d:s:n:h:m:", ["restart", "exc_site=", "bond_dim=", 
                                                                  "fmin=", "fmax=", "df=", 
                                                                  "num_exc_sites=", 
                                                                  "FCIDUMP=","SCRATCH=","eta=","molec="])
print('opts', opts, args)
for opt, arg in opts:
    if opt == '-r':
        restart = True
    elif opt == '-l':
        load_gs = True
    elif opt in ('-b', '--bond_dim'):
        bond_dim = int(arg)
    elif opt in ('-a', '--exc_site'):
        p = q = r = s = int(arg)
    elif opt in ('-p', '--p'):
        p = int(arg)
    elif opt in ('-q', '--q'):
        q = int(arg)
    elif opt in ('-r', '--r'):
        r = int(arg)
    elif opt in ('-s', '--s'):
        s = int(arg)
    elif opt in ('-f','--fmin'):
        fmin = float(arg)
    elif opt in ('-F','--fmax'):
        fmax = float(arg)
    elif opt in ('-d','--df'):
        df = float(arg)
    elif opt in ('-h','--eta'):
        eta = float(arg)
    elif opt in ('-n','--num_exc_sites'):
        num_exc_sites = int(arg)
    elif opt in ('--FCIDUMP'):
        fcidump_fstr = arg
    elif opt in ('-s', '--SCRATCH'):
        scratch_folder = arg
    elif opt in ('-m', '--molec'):
        molec = arg
    else:
        raise ValueError(f'{opt} is not a valid flag')

if __name__ == "__main__":

    print('molec', molec)
    if fmax is None:
        fmax = np.round((peak_loc_ev[molec] - 1) / ha2ev, 2)
        fmax = np.round((peak_loc_ev[molec]) / ha2ev + 2.0, 2)
        fmax = np.round((peak_loc_ev[molec]) / ha2ev + 10.0, 2)
        # fmax = np.round((peak_loc_ev[molec] + fmin_ev[molec]) / ha2ev, 2)
        fmax = float(fmax)
        # fmax = -91.0
        # fmax = -91.80
    else:
        fmax = np.round((peak_loc_ev[molec] + fmax) / ha2ev, 2)
        fmax = float(fmax)
    print('fmax', fmax, fmax * ha2ev)

    if fmin is None:
        fmin = np.round((peak_loc_ev[molec] + 2) / ha2ev, 2)
        fmin = np.round((peak_loc_ev[molec]) / ha2ev - 1.0, 2)
        fmin = np.round((peak_loc_ev[molec]) / ha2ev - 10.0, 2)
        # fmin = np.round((peak_loc_ev[molec] + fmax_ev[molec]) / ha2ev, 2)
        fmin = float(fmin)
        # fmin = -93.0
        # fmin = -91.60
    else:
        fmin = np.round((peak_loc_ev[molec] + fmin) / ha2ev, 2)
        fmin = float(fmin)
    print('fmin', fmin, fmin * ha2ev)

    if df is None:
        df = np.round((fmax - fmin) / 101, 3)
        # df = 0.001
        df = 0.0001

    main()
