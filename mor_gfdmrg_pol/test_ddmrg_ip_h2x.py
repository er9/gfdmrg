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
from run_mor import run_mor

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

## medium scale
fmin_ev = {'H2O': 0.0,
           'H2S': -30,
           'H2Se': -110.,
           'H2Te': -260.,
           'H2Po': -740.,
           }

fmax_ev = {'H2O': 0.0,
           'H2S': -20.,  # 10
           'H2Se': -95.,  # -70.,
           'H2Te': -240.,  # -220.,
           'H2Po': -720.,  # -680.,
           }

### fine scale
fmin_ev = {'H2O': 0.0,
           'H2S': -24,
           'H2Se': -103.5,  # -110.,
           'H2Te': -252.5,  # -260.,
           'H2Po': -731.,  # -740.,
           }

fmax_ev = {'H2O': 0.0,
           'H2S': -22.,
           'H2Se': -102.,  # -95.,  # -70.,
           'H2Te': -250.,  # -240.,  # -220.,
           'H2Po': -728.,  # -720.,  # -680.,
           }

#### parameters that can be set via flags
restart = False
load_gs = False
exc_site = None
eta = 0.02  # 0.005  # 0.02
num_exc_sites = 1
bond_dim = 300
scratch_folder = 'ip'
# fmin, fmax, df = -22.0, -18.0, 0.1
# fmin, fmax, df = -20.0, -19.25, 0.01
fmin, fmax, df = None, None, None
# fmin, fmax, df = -1, 1, 0.1
tol = 1e-5  # 0.001

molec = 'H2S'


def main(): 

    scratch_folder = f'mor_ip_no_{molec}_mpi1/eta{eta}_b{bond_dim}_v2/'
    fcidump_fstr = f'/global/homes/e/erikaye/gfdmrg/gfdmrg/ryanChalk/K_xanes/fcidump_{molec}/FCIDUMP'

    print('fs', fmin, fmax, df)
    print('exc site', exc_site, 'num sites', num_exc_sites)

    freq_str = f'fs_{fmin}_{fmax}_{df}'
    SCRATCH_GS = f"/pscratch/sd/e/erikaye/gfdmrg/{molec}/"
    SCRATCHDIR = f"/pscratch/sd/e/erikaye/gfdmrg/{scratch_folder}/{freq_str}/exc{exc_site}/"
    SCRATCHDIR_LDOS =  f"/pscratch/sd/e/erikaye/gfdmrg/{scratch_folder}/{freq_str}/ldos/"
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
                           symm_type=symm, mpi=True)
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

        driver.mpi.barrier()
        gs_ket = driver.copy_mps(gs_ket_, f"GS_KET_{exc_site}")
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
        driver.mpi.barrier()
        try:
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)
        except:
            restart_dir = SCRATCH_GS
            gs_tag = "GS_KET"
            if restart_dir != driver.scratch:
                for k in os.listdir(restart_dir):
                    if f'.{gs_tag}.' in k or k == f'{gs_tag}-mps_info.bin':
                        shutil.copy(restart_dir + "/" + k, driver.scratch + "/" + k)
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)

        gs_ket = driver.copy_mps(gs_ket, f"GS_KET_{exc_site}")
        driver.mpi.barrier()
        norm = np.abs(driver.expectation(gs_ket, impo, gs_ket)) / driver.mpi.size
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
    isites = range(gs_ket.n_sites) if exc_site is None else range(exc_site, exc_site + num_exc_sites)
    print('isites', isites)
    for isite in isites:

        print(f'excite {isite}', gs_ket.n_sites)
    
        if isite > gs_ket.n_sites:
            continue 

        try:
            raise IOError
            ldos = pickle.load(open(SCRATCHDIR_LDOS + f'a{isite}' + '.pkl','rb'))
            print('loaded ' + SCRATCHDIR_LDOS + f'a{isite}' )
            # print('ldos', ldos)

        except:

            dmpo = driver.get_site_mpo(op='D', site_index=isite, iprint=0)  ## electron excitation at site # isite
            dket = driver.get_random_mps(tag=f"DKET{exc_site}", bond_dim=bond_dim,
                                         center=gs_ket.center,
                                         target=dmpo.op.q_label + gs_ket.info.target)
            print('d multiply')
            driver.multiply(dket, dmpo, gs_ket, n_sweeps=10, # bond_dims=[bond_dim],
                            # thrds=[1E-10] * 10,
                            iprint=0)
            ### dket = dmpo * ket  (ket with removed electron)
            print('done d multiply', bond_dim)
            print('start mor')
            sys.stdout.flush()

            ### MOR sampling
         
            ldos = run_mor(driver, dket, mpo, dmpo, gs_ket, freqs, eta, tol, max_order=10, make_plot=True,
                            n_sweeps=20, 
                            bra_bond_dims=[bond_dim], ket_bond_dims=[bond_dim],
                            # noises=[1e-5] * 4 + [0],
                            # tol=1.0e-3,
                            # thrds=[1.0e-3] * 10,  # iprint=
                            reload=True, 
                            mpi_comm=comm,
                            )
            # print("FREQ = %8.2f GF[%d,%d] = %12.6f + %12.6f i" % (freq, isite, isite, gfmat[iw].real, gfmat[iw].imag))
            
            if driver.mpi.rank == 0:
                pickle.dump(ldos, open(SCRATCHDIR_LDOS + f'a{isite}' + '.pkl','wb'))

        if driver.mpi.rank == 0:
            plt.figure()
            plt.grid(which='major', axis='both', alpha=0.5)
            freqs = np.arange(fmin, fmax, df)
            plt.semilogy(freqs * -27.2, ldos, linestyle='-', marker='o', markersize=4,
                         mfc='white', mec="#7FB685", color="#7FB685")
            plt.xlabel("Frequency $\\omega$ [eV]")
            plt.ylabel("LDOS")
            plt.savefig(f'ldos_freqs_{freq_str}_eta{eta}_b{bond_dim}_i{exc_site}.png')
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
        exc_site = int(arg)
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
        fmax = np.round((peak_loc_ev[molec]) / ha2ev + 20.0, 2)
        fmax = np.round((peak_loc_ev[molec] + fmin_ev[molec]) / ha2ev, 2)
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
        fmin = np.round((peak_loc_ev[molec]) / ha2ev - 0.0, 2)
        fmin = np.round((peak_loc_ev[molec] + fmax_ev[molec]) / ha2ev, 2)
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
