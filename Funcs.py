import numpy as np
from scipy.integrate import simpson
import scipy.sparse as sparse

def norm(wf, dr, dR):
    return np.sum(wf * np.conj(wf)) * dr * dR


def normalize(wf, dx):
    """
    INPUT
    wf : A wavefunction (not a probability density)
    dx : The differential of space for the integration
    OUTPUT
    A : Normalization constant
    wf : Normalized wavefunction
    """
    A = simpson(wf * wf, dx=dx)
    return A, wf / np.sqrt(A)

def hFun(wavefun, H):
    return - H.dot(wavefun) * 1j

def rk4(wavefun, dt, H):
    """
    wavefun : A wavefunction (not a probability density)
    dt : Time step
    H : Matrix representation of the Hamiltonian in r representation
    """
    k1 = hFun(wavefun, H)
    k2 = hFun(wavefun + 0.5 * dt * k1, H)
    k3 = hFun(wavefun + 0.5 * dt * k2, H)
    k4 = hFun(wavefun + dt * k3, H)

    return wavefun + dt / 6 * (k1 + 2 * (k2 + k3) + k4)