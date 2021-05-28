import numpy as np
from scipy.special import erf

def sixth_laplacian(n, dx):
    lap = np.zeros((n, n))
    for i in range(n):
        lap[i,i] = -49 / 18
        if (i >= 1):
            lap[i, i-1] = 3 / 2
        if (i >= 2):
            lap[i, i-2] = -3 / 20
        if (i >= 3):
            lap[i, i-3] = 1 / 90
        if (n - i > 1):
            lap[i, i+1] = 3 / 2
        if (n - i > 2):
            lap[i, i+2] = -3 / 20
        if (n - i > 3):
            lap[i, i+3] = 1 / 90
    return lap / (dx * dx)


def kin(n, m, dx):
    """
    Returns the matrix representation of the kinetic operator
    n : Dimension of the space
    m : Mass of the particle
    dx : Self explanatory
    """
    return sixth_laplacian(n, dx) / (2 * m)


def softCoulomb():
    pass