import sys
import numpy as np
from scipy.sparse.linalg import gmres
from scipy.linalg import qr

def compute_gf(driver, dket, mpo, dmpo, ket, freq, eta, **dmrg_params):
    print('compute gf dmrg params', dmrg_params)
    bra = driver.copy_mps(dket, tag="BRA")  # initial guess
    gf_out = driver.greens_function(bra, mpo, dmpo, ket, freq, eta,
                                    **dmrg_params, 
                                    iprint=1)
                                    # n_sweeps=6, bra_bond_dims=[200], ket_bond_dims=[200],
                                    # noises=[1e-5] * 4 + [0],
                                    # thrds=[1E-6] * 10, iprint=2)
    print("FREQ = %8.5f GF = %12.6f + %12.6f i" % (freq, gf_out.real, gf_out.imag))

    return gf_out, bra


def mor_sampling(driver, gf_ket, mpo, exc_mpo, gs_ket, sample_freqs, eta, bra_exc_mpo=None,
                 V=None, orthogonalize=True, dmrg_opts=None, mpi_comm=None):
    """
    Moment matching model order reduction via sampling.

    Parameters:
    :param A: (numpy array) System matrix A.
    :param E: (numpy array) System matrix E.
    :param b: (numpy array) Input vector b.
    :param c: (numpy array) Output vector c.
    :param sample_freqs: (numpy array) Shifts for moment matching.
    :param V: (numpy array, optional) Exisiting subspace. Defaults to None.
    :param use_gmres: (bool, optional) Use GMRES for solving linear systems. Defaults to False.
    :param dmrg_opts: (dict, optional) GMRES parameters. Defaults to None.

    :returns
    At: (numpy array) Reduced system matrix A.
    Et: (numpy array) Reduced system matrix E.
    bt: (numpy array) Reduced input vector b.
    ct: (numpy array) Reduced output vector c.
    V : (numpy array) Orthogonal basis of the subspace.
    info (dict): Information about the GMRES iterations.
    """
    rank = mpi_comm.Get_rank() if mpi_comm is not None else 0
    size = mpi_comm.Get_size() if mpi_comm is not None else 1
    driver_size = 1 if driver.mpi is None else driver.mpi.size
    print('mpi comm', mpi_comm, 'rank', rank,'size',size)


    k = len(sample_freqs)
    dtype = complex if eta != 0.0 else sample_freqs.dtype

    if V is None:   ## list of TTs (subspace)
        V = []
    m = len(V)

    dmrg_opts = {} if dmrg_opts is None else dmrg_opts
    impo = driver.get_mpo(driver.expr_builder().add_term("", [], 1.0).iscale(1.0 / driver_size).finalize())


    # Construct subspace
    # V1 = np.zeros((n, k), dtype=dtype)
    for i in range(k):

        print('sample', sample_freqs[i])
        sys.stdout.flush()
        if driver.mpi is not None:
            driver.mpi.barrier()
        gf_val, gf_vec = compute_gf(driver, gf_ket, mpo, exc_mpo, gs_ket, sample_freqs[i], eta, **dmrg_opts)
        if driver.mpi is not None:
            driver.mpi.barrier()
        print('sample', sample_freqs[i], gf_val)
        print('rothogonalize', orthogonalize)
        sys.stdout.flush()

        if orthogonalize:
            print('here', len(V))
            for j, Vj in enumerate(V):
                print(f'orthogonalize new V({m + i}) wrt V({j})')
                if driver.mpi is not None:
                    driver.mpi.barrier()
                Vj = driver.copy_mps(Vj, tag='VJ')
                if driver.mpi is not None:
                    driver.mpi.barrier()
                # if driver.mpi.rank == 0:
                #     Vj = driver.copy_mps(Vj, tag='VJ')
                # driver.mpi.barrier()
                # sys.stdout.flush()
                # Vj = driver.load_mps(tag='VJ')
                # driver.mpi.barrier()

                print('start expec')
                ovlp = driver.expectation(Vj, impo, gf_vec)
                print('ovlp', ovlp)

                if driver.mpi is not None:
                    driver.mpi.barrier()
                orthog_vec = driver.copy_mps(gf_vec, tag=f'TMP0')
                if driver.mpi is not None:
                    driver.mpi.barrier()

                print('loaded, start addition')
                if driver.mpi is not None:
                    driver.mpi.barrier()
                sys.stdout.flush()

                Vj = driver.adjust_mps(Vj, dot=2)[0]
                if driver.mpi is not None:
                    driver.mpi.barrier()
                gf_vec = driver.adjust_mps(gf_vec, dot=2)[0]
                if driver.mpi is not None:
                    driver.mpi.barrier()
                orthog_vec = driver.adjust_mps(orthog_vec, dot=2)[0]
                if driver.mpi is not None:
                    driver.mpi.barrier()

                driver.addition(orthog_vec, ket_a=Vj, ket_b=gf_vec, mpo_a=-ovlp, mpo_b=1.0)

                if driver.mpi is not None:
                    driver.mpi.barrier()
                gf_vec = driver.copy_mps(orthog_vec, tag=f'TMP1')
                if driver.mpi is not None:
                    driver.mpi.barrier()

                # driver.mpi.barrier()
                # if driver.mpi.rank == 0:
                #     gf_vec = driver.copy_mps(orthog_vec, tag=f'TMP1')
                # driver.mpi.barrier()
                # gf_vec = driver.load_mps(tag=f'TMP1', nroots=1)
                # driver.mpi.barrier()

            sys.stdout.flush()

        if driver.mpi is not None:
            driver.mpi.barrier()

        print('here!')
        if driver.mpi is not None:
            driver.mpi.barrier()
        vec_norm = np.sqrt(driver.expectation(gf_vec, impo, gf_vec))
        print('vec norm', vec_norm)
        if driver.mpi is not None:
            driver.mpi.barrier()
        sys.stdout.flush()

        if driver.mpi is not None:
            driver.mpi.barrier()
        orthog_vec = driver.copy_mps(gf_vec, tag=f'ORTHOG_{m + i}')
        if driver.mpi is not None:
            driver.mpi.barrier()
        print('copied vec')

        # orthog_vec = driver.adjust_mps(orthog_vec, dot=2)[0]
        # driver.mpi.barrier()

        # scale_mpo = 1./vec_norm * driver.get_identity_mpo()
        scale_mpo = driver.get_mpo(driver.expr_builder().add_term("", [], 1.0/vec_norm).iscale(1.0 / driver_size).finalize())
        print('got mpo')
        if driver.mpi is not None:
            driver.mpi.barrier()
        # out_norm = driver.multiply(orthog_vec, driver.get_identity_mpo(), gf_vec) / driver.mpi.size / vec_norm
        out_norm = driver.multiply(orthog_vec, scale_mpo, gf_vec)
        print('out norm', out_norm, vec_norm)
        if driver.mpi is not None:
            driver.mpi.barrier()
            sys.stdout.flush()

        V += [orthog_vec]

        # w = np.linalg.solve(freqs[i] * E - A, b / np.linalg.norm(b))
        # V1[:, i] = w/np.linalg.norm(w)
        if mpi_comm is not None:
            mpi_comm.Barrier()
        if driver.mpi is not None:
            driver.mpi.barrier()
        sys.stdout.flush()

    if driver.mpi is not None:
        driver.mpi.barrier()

    # Construct reduced order model
    At = {}
    Et = {}
    bt = {}
    ct = {}

    # print('size?', driver.mpi.size, mpi_comm.Get_size())
    print('fill out rom matrices')

    ## fill out ROM matrices
    for j, Vj in enumerate(V):
       
        for i in range(k):
            # print('jj', j, driver.expectation(Vj, mpo, Vj))
            print('compute expec')
            sys.stdout.flush()
            At[(j, m + i)] = driver.expectation(Vj, mpo, V[m + i])
            print('At', j, m+i, At[(j,m+i)])
            if orthogonalize:
                # Et[(j, m + i)] = (1.0 if j == m+i else 0.0)  ## should be identity
                Et[(j, m + i)] = driver.expectation(Vj, impo, V[m + i])
            else:
                Et[(j, m + i)] = driver.expectation(Vj, impo, V[m + i])
            print('Et', j, m+i, Et[(j,m+i)])
        bt[(j,)] = driver.expectation(Vj, exc_mpo, gs_ket)  ## b = dmpo @ ket
        print('bt', j, bt[(j,)])
        # ct[(j,)] = driver.expectation(dket, impo, Vj)  ## c = dket
        bra_exc_mpo = exc_mpo if bra_exc_mpo is None else bra_exc_mpo
        ct[(j,)] = driver.expectation(gs_ket, bra_exc_mpo, Vj)  ## c (bra)
        print('ct', j, ct[(j,)])

    ## i don't know how to broadcast...
    if mpi_comm is not None:
        At = mpi_comm.bcast(At, root=0)
        Et = mpi_comm.bcast(Et, root=0)
        bt = mpi_comm.bcast(bt, root=0)
        ct = mpi_comm.bcast(ct, root=0)

    # At = V.T @ A @ V
    # Et = V.T @ E @ V
    # bt = V.T @ b
    # ct = V.T @ c

    # info = {}
    if driver.mpi is not None:
        driver.mpi.barrier()

    return At, Et, bt, ct, V, # info
