import numpy as np


def evaltf(A, E, b, c, omega):
    """
    Evaluate transfer function at specified frequencies.

    Parameters:
    A (numpy array): System matrix A.
    E (numpy array): System matrix E.
    b (numpy array): Input vector b.
    c (numpy array): Output vector c.
    omega (numpy array): Frequencies at which to evaluate the transfer function.

    Returns:
    y (numpy array): Transfer function values at frequencies omega.
    """
    k = len(omega)
    y = np.zeros(k, dtype=omega.dtype)
    for i in range(k):
        # Compute the transfer function at omega(i)
        y[i] = np.dot(c, np.linalg.solve(omega[i] * E + A, b))
    return y


def evaltf_new_c(driver, A, E, b, bra_exc_mpo, gs_ket, omega, V, num_sample: int =None, prev_ct=None):
    """
    :param A:
    :param E:
    :param b:
    :param bra_exc_mpo:
    :param gs_ket:
    :param omega:
    :param num_sample:
    :return:

    Computes spectrum (
    """
    if num_sample is None:
        num_sample = len(V)
    prev_num_sample = 0

    ct = np.empty((num_sample,), dtype=complex)
    if prev_ct is not None:
        m_, = prev_ct.shape
        ct[:m_] = prev_ct
        prev_num_sample = m_

    ## fill out ROM matrices
    for j, Vj in enumerate(V[:num_sample]):
        if j >= prev_num_sample:
            ct[(j,)] = driver.expectation(gs_ket, bra_exc_mpo, Vj)  ## c

    E = E[:num_sample, :num_sample]
    A = A[:num_sample, :num_sample]
    b = b[:num_sample]

    k = len(omega)
    y = np.zeros(k, dtype=omega.dtype)
    for i in range(k):
        # Compute the transfer function at omega(i)
        y[i] = np.dot(ct, np.linalg.solve(omega[i] * E + A, b))
    return y, ct
