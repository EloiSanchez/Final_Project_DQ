import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sparse
from scipy.integrate import simpson
from scipy.linalg import eigh
import Funcs as f
import operators as op

###############################################################################
#########################  Input & System parameters  #########################
###############################################################################

# Control parameters
numEigStates = 3 # Number of eigenstates to find
L = 19. # Distance between the fixed ions
M = 1863. # roton mass

# Electron grid
elecDx = 0.6
elecFactSpace = 6
elecDim = round(elecFactSpace * L / elecDx + 1)
elecSpace = np.linspace(-elecFactSpace * L / 2, elecFactSpace * L / 2, elecDim)
elecEye = np.identity(elecDim)

# Nucleus grid
nucDx = 0.06
nucFactSpace = 1.5
nucDim = round(nucFactSpace * L / nucDx + 1)
nucSpace = np.linspace(-nucFactSpace * L / 2, nucFactSpace * L / 2, nucDim)
nucEye = np.identity(nucDim)

# Interaction parameters
leftR = 4.4
rightR = 3.1
elecNucR = 7.

# Dynamic parameters
dt = 0.1  # Atomic units
tMax = 20  # Femtoseconds
AtomicToFs = 2.4189e-2
iMax = int(tMax / (dt * AtomicToFs))

print("==================  INITIAL  CONDITIONS  ==================")
print("\t ELECTRON SPACE")
print("\t  xmin = {:8.3f} \t dx = {:5.3f}".format(elecSpace[0], elecDx))
print("\t  xmax = {:8.3f} \t  N = {}".format(elecSpace[-1], elecDim))
print("\n\t NUCLUEAR SPACE")
print("\t  xmin = {:8.3f} \t dx = {:5.3f}".format(nucSpace[0], nucDx))
print("\t  xmax = {:8.3f} \t  N = {}".format(nucSpace[-1], nucDim))
print("\n\t DYNAMIC PARAMETERS")
print("\t  dt = {} a.u.    total time = {} fs".format(dt, tMax))
print("\t  number of iterations = {}".format(iMax))
print("\n\t INTERACTION PARAMETERS")
print("\t  Rl = {}\tRr = {}\tRf = {}".format(leftR, rightR, elecNucR))
print("===========================================================\n")

###############################################################################

###############################################################################
####################  Finding the potential in full-space  ####################
###############################################################################
print("=======================  STATICS  =========================")

print("\n\tCalculating the full-space potential and plot.")
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

###############################################################################

###############################################################################
################## Diagonalization of electronic hamiltonian ##################
###############################################################################

print("\n\tDiagonalizing electronic hamiltonian.")

potLeft = np.diag(op.softCoulomb(elecSpace, -L/2, leftR))
potRight = np.diag(op.softCoulomb(elecSpace, L/2, rightR))

elecHam = op.kin(elecDim, 1, elecDx) - potLeft - potRight # + potential operators to be done

elecHamR = []
elecEigenstates = np.zeros((elecDim, numEigStates, nucDim))
elecEigenvalues = np.zeros((numEigStates, nucDim))

for i in range(nucDim):
    elecHamR.append(elecHam - np.diag(op.softCoulomb(elecSpace, nucSpace[i], elecNucR)))
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
# axEignStates = Axes3D(figEign)
axEignStates = figEign.add_subplot(122, projection='3d')

elecEigenvalues += 1 / np.abs(nucSpace - L/2) + 1 / np.abs(nucSpace + L/2)
# cmaps = ['YlGnBu', 'YlOrRd', 'YlGn']
cmaps = ['winter', 'autumn', 'summer']
for i in range(numEigStates):
    axEignVals.plot(nucSpace, elecEigenvalues[i,:], label="state {}".format(i))
    axEignStates.plot_surface(elecMeshGrid, nucMeshGrid, elecEigenstates[:,i,:]**2 + 0.3*i, \
        label=r"$|\Psi_{}|^2$".format(i), ccount=25 , cmap=cmaps[i])

axEignVals.set_ylim((-0.25,0.1))
axEignVals.legend()
axEignVals.set_title("BOPES")
axEignVals.set_xlabel(r"$R$ (a.u.)")
axEignVals.set_ylabel("Energy (a.u.)")

# axEignStates.legend()
axEignStates.set_title(r"Eigenstates")
axEignStates.set_zlabel(r"Probability densities (Bohr$^{-1}$)")
axEignStates.set_xlabel(r"$r$ (a.u.)")
axEignStates.set_ylabel(r"$R$ (a.u.)")
# axEignStates.set_xtics(15)
# axEignStates.set_ytics(5)
# axEignStates.set_xlim3d((-20, 20))
axEignStates.view_init(elev=10., azim=295)

# plt.show()

###############################################################################

###############################################################################
########################### Calculation of the NACS ###########################
###############################################################################

print("\n\tCalculating non-adiabatic couplings.")

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
# figFullPot.savefig("FullPotential.png", dpi=600)
# figEign.savefig("Eigen.png", dpi=600)
# plt.show()

print("\n===========================================================\n")

###############################################################################

###############################################################################
################################### Dynamics ##################################
###############################################################################

print("=======================  DYNAMICS  ========================")

rn0 = -7.
sigma = 1. / np.sqrt(2.85)
phi_n = np.sqrt(np.sqrt(2) / (sigma * np.sqrt(np.pi))) * np.exp(-(nucSpace - rn0)**2 / sigma**2)  # vector R elements

phi_e = elecEigenstates[:,1,:]  # Dimensio rxR

phi_ini = phi_n * phi_e  # Dimensio rxR

phi = phi_ini.flatten()  # vector de r*R elements

# Kinetic part
hamiltonian = sparse.kron(op.kin(elecDim, 1, elecDx), nucEye) + sparse.kron(elecEye, op.kin(nucDim, M, nucDx))

# Electron - proton Interaction
hamiltonian -= sparse.diags(op.softCoulomb(elecMeshGrid, nucMeshGrid, elecNucR).flatten())

# Electron - fixed ion Interaciton
hamiltonian -= sparse.diags(np.transpose(np.broadcast_to(op.softCoulomb(elecSpace, -L/2, leftR) + \
                op.softCoulomb(elecSpace, L/2, rightR), (nucDim, elecDim))).flatten())
plt.show()

t = 0
phiSave = np.zeros_like(phi)
# Get time 0
elecRedProb, nucRedProb = f.getRedProbs(phi, elecDim, nucDim, elecDx, nucDx)
nucRedProbAll = [nucRedProb]
elecRedProbAll = [elecRedProb]
timeAll = [0]

for i in range(iMax):
    if i % 10 == 0: print("\tCurrent state {:.3f} %".format(i / iMax * 100), end="\r", flush=True)
    phiNew = f.rk4(phi, dt, hamiltonian)
    phi = np.copy(phiNew)
    # nucFuncs = f.getBO(phi, elecEigenstates, elecDx)
    # if (i + 1) % 1000 == 0:
    #     print("At time {} fs we are in iter {}".format(t * AtomicToFs,i))
    #     elecRedProb, nucRedProb = f.getRedProbs(phi, elecDim, nucDim, elecDx, nucDx)
    #     print(np.sum(elecRedProb)*elecDx, np.sum(nucRedProb)*nucDx)
    #     plt.plot(nucSpace, np.real(nucRedProb), label="nuc")
    #     plt.plot(elecSpace, np.real(elecRedProb), label="elec")
    #     plt.legend()
    #     plt.xlim(-10, 10)
    #     plt.show()
    #     phiNorm = f.norm(phi, elecDx, nucDx)
    #     print(phiNorm)
    #     print(np.sum(elecSpace*elecRedProb)*elecDx)
    #     print(np.sum(nucSpace*nucRedProb)*nucDx)
    #     # print((np.real(np.conj(phiSave)*phiSave)-np.real(np.conj(phi)*phi)).reshape(elecDim,nucDim)[:,10])
    #     phiSave = phi.copy()
    if (i + 1) % 25 == 0:
        t = dt * (i + 1)
        elecRedProb, nucRedProb = f.getRedProbs(phi, elecDim, nucDim, elecDx, nucDx)
        nucRedProbAll.append(nucRedProb)
        elecRedProbAll.append(elecRedProb)
        timeAll.append(t * AtomicToFs)

###############################################################################

###############################################################################
################################## Animation ##################################
###############################################################################

figAnim = plt.figure()
axAnim = figAnim.add_subplot()
elecAnim = axAnim.plot(elecSpace, elecRedProbAll[0], label=r"$\rho_e(r)$")
nucAnim = axAnim.plot(nucSpace, nucRedProbAll[0], label=r"$\rho_N(R)$")
axAnim.set_title("Time = {:.1f} fs".format(timeAll[0]))
axAnim.legend()

def animate(i, elecProb, nucProb, time):
    elecAnim[0].set_ydata(elecProb[i])
    nucAnim[0].set_ydata(nucProb[i])
    axAnim.set_title("Time = {:.1f} fs".format(time[i]))

animation = FuncAnimation(
    figAnim, 
    animate, 
    frames = int(iMax / 25), 
    fargs = (elecRedProbAll, nucRedProbAll, timeAll),
    interval = 33.3
    )

animation.save(
    "Results.mp4", 
    dpi = 300, 
    progress_callback = lambda j, n: print(f'\tSaving frame {j} of {n}', end="\r", flush=True)
    )

plt.show()