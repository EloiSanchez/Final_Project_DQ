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
    return - sixth_laplacian(n, dx) / (2 * m)


def softCoulomb(pos1, pos2, rC):
    """
    Returns the soft-core Coulomb interaction energy of two ions parametrized 
    by rC. If pos1 is a vector it returns the matrix representation, if it is a
    matrix (coming from a meshgrid) it returns the potential evaluated at all
    points in the pos1 matrix.
    pos1 : (array) Positions to evaluate the "moving" ion
    pos2 : (float) Position of the fixed ion
    rC : (float) Paramater of the soft-core Coulomb potential
    """
    x = np.abs(pos1 - pos2) / rC
    return np.where(x <= 1e-6,                          # Condition
        2 * (1 - (x ** 2) / 3) / (rC * np.sqrt(np.pi)),   # If True
        erf(x) / (rC * x))                                # If False