import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import helperFuncs as hf
import operators as op

# System parameters
L = 19. # Distance between the fixed ions

# Electron grid
eDim = 100
eSpace = np.linspace(-L/2, L/2, eDim)
eEye = np.identity(eDim)

# Nucleus grid
nDim = 21
nSpace = np.linspace(-L/2, L/2, nDim)
nEye = np.identity(nDim)


######################################
########### Laplacian test ###########
######################################

y = np.sin(eSpace)
ddy = np.matmul(op.sixth_laplacian(eDim, eSpace[1] - eSpace[0]), y)
anal = -np.sin(eSpace)

plt.plot(eSpace, y, label="func")
plt.plot(eSpace, ddy, label="ddy")
plt.plot(eSpace, anal, label="analytical")

plt.legend()
plt.show()