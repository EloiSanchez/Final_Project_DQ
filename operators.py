import numpy as np

def sixth_laplacian(n, m, dx):
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
    return -0.5 * lap / (m * dx * dx)
