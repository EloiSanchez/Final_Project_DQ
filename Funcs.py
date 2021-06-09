import numpy as np
from scipy.integrate import simpson
import scipy.sparse as sparse

def norm(wf, dr, dR):
    """
    wf : Wavefunction in full-space
    dr : Space step for the electron
    dR : Space step for the nucleus
    return : Scalar with the norm
    """
    return np.sum(wf * np.conj(wf)) * dr * dR

def normalize(wf, dx):
    """
    wf : A wavefunction (not a probability density)
    dx : The differential of space for the integration
    return : (A : Normalization constant, wf : Normalized wavefunction)
    """
    A = simpson(wf * wf, dx=dx)
    return A, wf / np.sqrt(A)

def hFun(wavefun, H):
    return - H.dot(wavefun) * 1j

def rk4(wavefun, dt, H):
    """
    wavefun : A wavefunction (not a probability density)
    dt : Time step
    H : Matrix representation of the Hamiltonian in space representation
    return : wave function at next time-step
    """
    k1 = hFun(wavefun, H)
    k2 = hFun(wavefun + 0.5 * dt * k1, H)
    k3 = hFun(wavefun + 0.5 * dt * k2, H)
    k4 = hFun(wavefun + dt * k3, H)

    return wavefun + dt / 6 * (k1 + 2 * (k2 + k3) + k4)

def getBO(wavefun, elecStates, dr):
    """
    work in progress
    """
    nucStates = np.zeros_like(elecStates)
    return nucStates

def getRedProbs(wavefun, elecDim, nucDim, dr, dR):
    """
    Get reduced probabilities
    wavefun : The full-space wavefunction
    elecDim, nucDim : Number of gridpoints for elec and nuc
    dr, dR : Space steps in electron and nuclear space
    """
    dens = np.conj(wavefun) * wavefun
    dens = dens.reshape((elecDim, nucDim))
    redProbElec = np.sum(dens, axis=1) * dR
    redProbNuc = np.sum(dens, axis=0) * dr
    return redProbElec, redProbNuc