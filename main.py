import numpy as np
import matplotlib.pyplot as plt
from numpy.core import numeric
from numpy.core.numeric import full
from scipy.integrate import simpson
from scipy.linalg import eigh
import helperFuncs as hf
import operators as op

#################################################################
##################  Input & System parameters  ##################
#################################################################

# Control parameters
numEigStates = 3 # Number of eigenstates to find
L = 19. # Distance between the fixed ions
M = 1863. #Proton mass

# Electron grid
elecDim = 401
elecSpace = np.linspace(-L, L, elecDim)
elecDx = elecSpace[1] - elecSpace[0]
elecEye = np.identity(elecDim)

# Nucleus grid
nucDim = 201
nucSpace = np.linspace(-L/2, L/2, nucDim)
nucDx = nucSpace[1] - nucSpace[0]
nucEye = np.identity(nucDim)

# Interaction parameters
leftR = 4.4
rightR = 3.1
elecNucR = 7.

#################################################################

#################################################################
#############  Finding the potential in full-space  #############
#################################################################

# Obtain X and Y
[nucMeshGrid, elecMeshGrid] = np.meshgrid(nucSpace, elecSpace)  

# Obtain Z
fullSpacePot = - op.softCoulomb(elecMeshGrid, nucMeshGrid, elecNucR) \
    - op.softCoulomb(elecMeshGrid, -L/2, leftR) \
    - op.softCoulomb(elecMeshGrid, L/2, rightR) \
    + 1 / np.abs(nucMeshGrid - L/2) + 1 / np.abs(nucMeshGrid + L/2)

# Create figure
figFullPot = plt.figure(figsize=(6.4*1.2,4.8))
axFullPot = figFullPot.add_subplot()

# Format Figure
contourLims = [-0.4, 0.4]  # zmax and zmin
axFullPot.set_xlim(-8.2,5) 
axFullPot.set_ylim(-16,16)

axFullPot.set_title("Potential Energy Surface in full-space")
axFullPot.set_xlabel(r"$R$ (a.u.)")
axFullPot.set_ylabel(r"$r$ (a.u.)")

colorMap = "viridis" # Check https://tinyurl.com/3a4y73kj for more

# Plot figure
imf = axFullPot.contourf(nucSpace, elecSpace, fullSpacePot, cmap=colorMap,\
    vmin=contourLims[0], vmax=contourLims[1], \
    levels=np.linspace(contourLims[0], contourLims[1], 18))

imc = axFullPot.contour(nucSpace, elecSpace, fullSpacePot, colors="black",\
    linewidths=0.5, linestyles="solid", \
    levels=np.linspace(contourLims[0], contourLims[1], 18))

colorBar = figFullPot.colorbar(imf, format="%.2f")

# plt.show()

#################################################################

##################################################################
######################### Laplacian test #########################
##################################################################

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

##################################################################

#################################################################
########### Diagonalization of electronic hamiltonian ###########
#################################################################

potLeft = op.softCoulomb(elecSpace, -L/2, leftR)
potRight = op.softCoulomb(elecSpace, L/2, rightR)

elecHam = op.kin(elecDim, 1, elecDx) - potLeft - potRight # + potential operators to be done

elecHamR = []
elecEigenstates = np.zeros((elecDim, numEigStates, nucDim))
elecEigenvalues = np.zeros((numEigStates, nucDim))

for i in range(nucDim):
    elecHamR.append(elecHam - op.softCoulomb(elecSpace, nucSpace[i], elecNucR))
    elecEigenvalues[:,i], elecEigenstates[:,:,i] = eigh(elecHamR[i], subset_by_index=[0, numEigStates - 1])
    for j in range(numEigStates):
        elecEigenstates[:,j,i] /= np.sqrt(simpson(elecEigenstates[:,j,i] * elecEigenstates[:,j,i], dx=elecDx))

limit = 0.05
for i in range(1, nucDim):
    for j in range(numEigStates):
        aux = np.sum(elecEigenstates[:,j,i-1]*elecEigenstates[:,j,i])*elecDx
        if aux<limit:
            elecEigenstates[:,j,i]*=-1


figEign = plt.figure(figsize=(6.4*2, 4.8))
axEignVals = figEign.add_subplot(121)
axEignStates = figEign.add_subplot(122)

elecEigenvalues += 1 / np.abs(nucSpace - L/2) + 1 / np.abs(nucSpace + L/2)

for i in range(numEigStates):
    axEignVals.plot(nucSpace, elecEigenvalues[i,:], label="state {}".format(i))
    axEignStates.plot(elecSpace, elecEigenstates[:,i,nucDim//2]**2, label=r"$|\Psi_{}|^2$".format(i))

axEignVals.set_ylim((-0.25,0.1))
axEignVals.legend()
axEignVals.set_title("BOPES")
axEignVals.set_xlabel(r"$R$ (a.u)")
axEignVals.set_ylabel("Energy (a.u.)")

axEignStates.legend()
axEignStates.set_title(r"Eigenstates with $R={}$".format(nucSpace[nucDim//2]))
axEignStates.set_ylabel(r"Probability densities (Bohr$^-1$)")
axEignStates.set_xlabel(r"$r$ (a.u)")

# plt.show()

#################################################################

#################################################################
#################### Calculation of the NACS ####################
#################################################################

NACs = np.zeros((numEigStates, numEigStates, nucDim))

axNAC = axEignVals.twinx()
colors=["violet", "mediumorchid", "indigo"]

iAux = 0
for i in range(numEigStates):
    lapElec = np.matmul(op.kin(nucDim, M, nucDx), np.transpose(elecEigenstates[:, i, :]))
    for j in range(i, numEigStates):
        for k in range(nucDim):
            NACs[i, j, k] = simpson(elecEigenstates[:, j, k] * lapElec[k, :], dx=elecDx)
        NACs[j, i, :] = NACs[i, j, :]
        if i != j:
            axNAC.plot(nucSpace, NACs[i, j, :], color=colors[iAux], label="NAC {}-{}".format(i,j))
            iAux += 1

axNAC.legend()
axNAC.set_ylabel(r"NACs (Bohr$^-1$)")

plt.tight_layout()
plt.show()

#################################################################