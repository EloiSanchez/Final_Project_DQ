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
    return np.real(np.sum(wf * np.conj(wf)) * dr * dR)
    
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

def getBO(wavefun, elecStates, numEigstates, nucDim, elecDim, dr):
    """
    wavefun : The full-space wavefunction
    elecStates : Electronic eigenstates considered
    numEigstates : nº of states considered
    nucDim, elecDim : Number of gridpoints for elec and nuc
    dr : Space step in electron space
    """
    wavefunMat = wavefun.reshape((elecDim, nucDim))
    nucStates = np.zeros((nucDim, numEigstates), dtype=np.complex64)
    for i in range(numEigstates):
        nucStates[:,i] = np.sum(np.conj(elecStates[:,i,:] * wavefunMat), axis=0) * dr
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
    redProbElec = np.real(np.sum(dens, axis=1) * dR)
    redProbNuc = np.real(np.sum(dens, axis=0) * dr)
    return redProbElec, redProbNuc

def getDecDyn(nucStates, N, dR):
    """
    Get elements of the upper triangular matrix as (ex. with N=3) 11 12 13 22 23 33
    nucstates : Nuclear wavefunctions
    N : Number of eigenstates
    dR : Space step in nuclear space
    """
    DecDyn = []
    for i in range(N):
        for j in range(i + 1, N):
            DecDyn.append(np.abs(np.sum((np.conj(nucStates[:,i]) * nucStates[:,j])) * dR))
    return DecDyn