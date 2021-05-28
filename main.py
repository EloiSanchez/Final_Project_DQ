import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import eigh
import helperFuncs as hf
import operators as op

## System parameters
# Control parameters
numEigStates = 3 # Number of eigenstates to find
L = 19. # Distance between the fixed ions

# Electron grid
elecDim = 201
elecSpace = np.linspace(-L/2, L/2, elecDim)
elecDx = elecSpace[1] - elecSpace[0]
elecEye = np.identity(elecDim)

# Nucleus grid
nucDim = 201
nucSpace = np.linspace(-L/2, L/2, nucDim)
nucDx = nucSpace[1] - nucSpace[0]
nucEye = np.identity(nucDim)

######################################
########### Laplacian test ###########
######################################

# y = np.sin(elecSpace)
# ddy = np.matmul(op.sixth_laplacian(elecDim, elecSpace[1] - elecSpace[0]), y)
# anal = -np.sin(elecSpace)

# fig = plt.figure()
# ax = fig.add_subplot()

# ax.plot(elecSpace, y, label="func")
# ax.plot(elecSpace, ddy, label="ddy")
# ax.plot(elecSpace, anal, label="analytical")

# ax.legend()

# fig.savefig("foto.png")

#####################################

potLeft = op.softCoulombFixed(elecSpace, -L/2, 3.1)
potRight = op.softCoulombFixed(elecSpace, L/2, 4.4)

elecHam = op.kin(elecDim, 1, elecDx) - potLeft - potRight # + potential operators to be done

elecHamR = []
elecEigenstates = np.zeros((elecDim, numEigStates, nucDim))
elecEigenvalues = np.zeros((numEigStates, nucDim))

for i in range(nucDim):
    elecHamR.append(elecHam - op.softCoulombFixed(elecSpace, nucSpace[i], 7))
    elecEigenvalues[:,i], elecEigenstates[:,:,i] = eigh(elecHamR[i], subset_by_index=[0, numEigStates - 1])

fig3 = plt.figure()
ax3 = fig3.add_subplot()

elecEigenvalues += 1 / np.abs(nucSpace - L/2) + 1 / np.abs(nucSpace + L/2)

for i in range(numEigStates):
    ax3.plot(nucSpace, elecEigenvalues[i,:], label="state {}".format(i))


# ax3.set_xlim((-8,8))
ax3.set_ylim((-0.25,0.1))
ax3.legend()

#####################################
# Per veure els eigenstates descomentar aquest codi
# fig4 = plt.figure()
# ax4 = fig4.add_subplot()
# ax4.plot(elecSpace, np.abs(elecEigenstates[:,0,1]), label="state 1")
# ax4.plot(elecSpace, np.abs(elecEigenstates[:,1,1]), label="state 2")
# ax4.plot(elecSpace, np.abs(elecEigenstates[:,2,1]), label="state 3")
# ax4.legend()
#####################################

plt.show()