import numpy as np
from scipy.integrate import simpson

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