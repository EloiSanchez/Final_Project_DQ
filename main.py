import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import helperFuncs as hf
import operators as op

# System parameters
L = 19. # Distance between the fixed ions

# Electron grid
elecDim = 101
elecSpace = np.linspace(-L/2, L/2, elecDim)
elecDx = elecSpace[1] - elecSpace[0]
elecEye = np.identity(elecDim)

# Nucleus grid
nucDim = 21
nucSpace = np.linspace(-L/2, L/2, nucDim)
nucDx = nucSpace[1] - nucSpace[0]
nucEye = np.identity(nucDim)


######################################
########### Laplacian test ###########
######################################

y = np.sin(elecSpace)
ddy = np.matmul(op.sixth_laplacian(elecDim, elecSpace[1] - elecSpace[0]), y)
anal = -np.sin(elecSpace)

plt.plot(elecSpace, y, label="func")
plt.plot(elecSpace, ddy, label="ddy")
plt.plot(elecSpace, anal, label="analytical")

plt.legend()
plt.show()

######################################

eHam = op.kin(elecDim, 1, elecDx)  # + potential operators to be done
