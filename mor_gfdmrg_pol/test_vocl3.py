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



#### parameters that can be set via flags
restart = False
load_gs = False
exc_site = 0
eta = 0.001  # 0.02  # 0.02
num_exc_sites = 1
bond_dim = 300
scratch_folder = f'mor_ip_no_mpi{size}'  # 'mor_ip_grid3_o2'
# scratch_folder = f'mor_ip_no_mpi{size}_v2'  # n_sweeps=20 -- too slow.
# fmin, fmax, df = -22.0, -18.0, 0.1
# fmin, fmax, df = -20.0, -19.25, 0.01
fmin, fmax, df = -20.0, -19.20, 0.01
fmin, fmax, df = -20.0, -19.20, 0.001
# fmin, fmax, df = -1, 1, 0.1
tol = 0.001

fcidump_fstr = '/global/homes/e/erikaye/gfdmrg/gfdmrg/FCIDUMP/FCIDUMP.fcidump'  ## 46 sites # 'FCIDUMP.Ga'

def main(): 

    print('bond dim', bond_dim, 'eta', eta)

    print('exc site', exc_site, 'num sites', num_exc_sites)
    freq_str = f'fs_{fmin}_{fmax}_{df}'
    SCRATCH_GS = f"/pscratch/sd/e/erikaye/gfdmrg/vocl3/"
    SCRATCHDIR = f"/pscratch/sd/e/erikaye/gfdmrg/{scratch_folder}/{freq_str}/eta{eta}_b{bond_dim}/exc{exc_site}/"
    SCRATCHDIR_LDOS =  f"{SCRATCHDIR}/ldos/"
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
    
    symm = SymmetryTypes.SGFCPX
    
    driver = SOCDMRGDriver(scratch=SCRATCHDIR, # n_threads=64, 
                           symm_type=symm, mpi=True)
    # driver = SOCDMRGDriver(scratch=SCRATCHDIR, symm_type=symm, mpi=True)
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
        
        print('computing ground state')
        gs_ket_ = driver.get_random_mps(tag=f"KET", bond_dim=bond_dim)
        gs_energy = driver.dmrg(mpo, gs_ket_, n_sweeps=6, # bond_dims=bond_dims, noises=noises,
                                thrds=thrds, iprint=1)

        print('Ground state energy = %20.15f' % gs_energy)

        driver.mpi.barrier()
        gs_ket = driver.copy_mps(gs_ket_, f"GS_KET_{exc_site}")
        driver.mpi.barrier()

    else:
        driver.mpi.barrier()
        gs_tag = "GS_KET"
        try:
            # raise IOError
            print('try loading')
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)
        except:
            if driver.mpi.rank == 0:
                print('try loading from ', SCRATCH_GS)
                restart_dir = SCRATCH_GS
                if restart_dir != driver.scratch:
                    for k in os.listdir(restart_dir):
                        if f'.{gs_tag}.' in k or k == f'{gs_tag}-mps_info.bin':
                            shutil.copy(restart_dir + "/" + k, driver.scratch + "/" + k)
                print('gs tag', gs_tag)
            driver.mpi.barrier()
            gs_ket = driver.load_mps(tag=gs_tag, nroots=1)

        gs_ket = driver.copy_mps(gs_ket, f"GS_KET_{exc_site}")
        driver.mpi.barrier()

        gs_ket = driver.adjust_mps(gs_ket, dot=2)[0]
        driver.mpi.barrier()

        norm = np.abs(driver.expectation(gs_ket, impo, gs_ket)) / driver.mpi.size
        gs_energy = driver.expectation(gs_ket, mpo, gs_ket)
        print('<x|mpo|x>', gs_energy, 'ovlp', norm)
        print(f'Ground state energy = {gs_energy/norm}')
        print('driver rank', driver.mpi.rank)
        print('driver size', driver.mpi.size)

    

    ###################
    ## D-DMRG
    ###################
    print('start dynamical DMRG')

    ## update scratch to avoid overwriting?  ## messes up gs_ket calling then...
    # driver.scratch = SCRATCHDIR_LDOS

    mpo.const_e -= gs_energy
    # eta = 0.1  ## TlH   (examples tend to use 0.0005)
    print('eta', eta)

    isite = exc_site
    freqs = np.arange(fmin, fmax, df)

    print(f'excite {isite}')

    if isite > 46: # gs_ket.n_sites:
        raise ValueError

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
     
        ldos = run_mor(driver, dket, mpo, dmpo, gs_ket, freqs, eta, tol, make_plot=True,
                        n_sweeps=6, bra_bond_dims=[bond_dim], ket_bond_dims=[bond_dim],
                        # noises=[1e-5] * 4 + [0],
                        # tol=1.0e-3,
                        # thrds=[1.0e-3] * 10,  # iprint=
                        # reload=True, 
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



#################################
## set parameters via input flags
#################################


opts, args = getopt.getopt(sys.argv[1:], "rla:b:f:F:d:s:n:h:", ["restart", "exc_site=", "bond_dim=", 
                                                                "fmin=", "fmax=", "df=", 
                                                                "num_exc_sites=", 
                                                                "FCIDUMP=","SCRATCH=","eta="])
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
    else:
        raise ValueError(f'{opt} is not a valid flag')

if __name__ == "__main__":
    main()
