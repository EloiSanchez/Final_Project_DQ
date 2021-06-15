import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from numpy.core import numeric
import os
import scipy.sparse as sparse
from scipy.integrate import simpson
from scipy.linalg import eigh
import Funcs as f
import operators as op
from input import *

###############################################################################
#########################  Input & System parameters  #########################
###############################################################################

# Check if there are any problems for directory results and create it
if os.path.exists(resultsDir):
    dirControl = True
    while dirControl:
        choice = input("""
The directory {} already exists. I want to... 
    ...rename the results directory (R)
    ...overwrite the existing directory (O)
    ...exit program (X)
>> """.format(resultsDir)).lower().strip()
        if choice.startswith('r'):
            resultsDir = input('\nIntroduce new directory >> ')
            os.makedirs(resultsDir)
            dirControl = False
        elif choice.startswith('o'):
            print('\nOverwriting existing directory.')
            dirControl = False
        elif choice.startswith('x'):
            print('\nNORMAL TERMINATION\n')
            quit()
else:
    os.makedirs(resultsDir)

# Electron grid
elecDim = round(elecFactSpace * L / elecDx + 1)
elecSpace = np.linspace(-elecFactSpace * L / 2, elecFactSpace * L / 2, elecDim)
elecEye = np.identity(elecDim)
elecDx = elecSpace[1] - elecSpace[0]

# Nucleus grid
nucDim = round(nucFactSpace * L / nucDx + 1)
nucSpace = np.linspace(-nucFactSpace * L / 2, nucFactSpace * L / 2, nucDim)
nucEye = np.identity(nucDim)
nucDx = nucSpace[1] - nucSpace[0]

# Dynamic parameters
iMax = int(tMax / (dt * AtomicToFs))

print("\n==================  INITIAL  CONDITIONS  ==================")
print("\t ELECTRON SPACE")
print("\t  xmin = {:8.3f} \t dx = {:7.5f}".format(elecSpace[0], elecDx))
print("\t  xmax = {:8.3f} \t  N = {}".format(elecSpace[-1], elecDim))
print("\n\t NUCLEAR SPACE")
print("\t  xmin = {:8.3f} \t dx = {:7.5f}".format(nucSpace[0], nucDx))
print("\t  xmax = {:8.3f} \t  N = {}".format(nucSpace[-1], nucDim))
print("\n\t DYNAMIC PARAMETERS")
print("\t  dt = {} a.u.    total time = {} fs".format(dt, tMax))
print("\t  number of iterations = {}".format(iMax))
print("\n\t INTERACTION PARAMETERS")
print("\t  Rl = {}\tRr = {}\tRf = {}".format(leftR, rightR, elecNucR))
print("===========================================================\n")

np.seterr(all='ignore')

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

figEignVals = plt.figure()
axEignVals = figEignVals.add_subplot(111)
figEignStates = plt.figure(figsize=(6.4, 4.8*1.5))
axEignStates = [figEignStates.add_subplot(11 + 100*numEigStates + i) for i in range(numEigStates)]

elecEigenvalues += 1 / np.abs(nucSpace - L/2) + 1 / np.abs(nucSpace + L/2)
cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Yellows', 'Greys']
imEigs = []
for i in range(numEigStates):
    axEignVals.plot(nucSpace, elecEigenvalues[i,:], label=r"$\varphi_{}$".format(i))
    imEigs.append(axEignStates[i].contourf(nucMeshGrid, elecMeshGrid, elecEigenstates[:,i,:]**2, \
    cmap=cmaps[i]))
    axEignStates[i].contour(nucMeshGrid, elecMeshGrid, elecEigenstates[:,i,:]**2,
                            colors="black", linewidths=0.3, linestyles="solid")
[figEignStates.colorbar(imEigs[i], format="%.2f", ax=axEignStates[i]) for i in range(numEigStates)]

axEignVals.set_ylim((-0.25,0.1))
axEignVals.legend()
axEignVals.set_title("BOPES")
axEignVals.set_xlabel(r"$R$ (a.u.)")
axEignVals.set_ylabel("Energy (a.u.)")

[axEignStates[i].set_title(r"Eigenstate $|\varphi_{}|^2$".format(i)) for i in range(numEigStates)]
axEignStates[-1].set_xlabel(r"$R$ (a.u.)")
[axEignStates[i].set_ylabel(r"$r$ (a.u.)") for i in range(numEigStates)]
[axEignStates[i].set_ylim((-1.5 * L/2, 1.5 * L/2)) for i in range(numEigStates)]

plt.tight_layout()
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
        NACs[j, i, :] = - NACs[i, j, :]
        if i != j:
            axNAC.plot(nucSpace, NACs[i, j, :], color=colors[iAux], label="NAC {}-{}".format(i,j))
            iAux += 1

axNAC.legend()
axNAC.set_ylabel(r"NACs (Bohr$^-1$)")

plt.tight_layout()
figFullPot.savefig(resultsDir + "/FullPotential.png", dpi=600)
figEignVals.savefig(resultsDir + "/EigenVals.png", dpi=600)
figEignStates.savefig(resultsDir + "/EigenStates.png", dpi=600)
# plt.show()

print("\n===========================================================\n")

###############################################################################

###############################################################################
################################### Dynamics ##################################
###############################################################################

print("=======================  DYNAMICS  ========================\n")

phi_n = np.sqrt(np.sqrt(2) / (sigma * np.sqrt(np.pi))) * np.exp(-(nucSpace - rn0)**2 / sigma**2)  # vector R elements

phi_e = elecEigenstates[:,initEigenstate,:]  # Dimensio rxR

phi_ini = phi_n * phi_e  # Dimensio rxR

phi = phi_ini.flatten()  # vector de r*R elements

# Kinetic part
hamiltonian = sparse.kron(op.kin(elecDim, 1, elecDx), nucEye) + sparse.kron(elecEye, op.kin(nucDim, M, nucDx))

# Electron - proton Interaction
hamiltonian -= sparse.diags(op.softCoulomb(elecMeshGrid, nucMeshGrid, elecNucR).flatten())

# Electron - fixed ion Interaciton
hamiltonian -= sparse.diags(np.transpose(np.broadcast_to(op.softCoulomb(elecSpace, -L/2, leftR) + \
                op.softCoulomb(elecSpace, L/2, rightR), (nucDim, elecDim))).flatten())

hamiltonian += sparse.diags(np.broadcast_to(1 / np.abs(nucSpace - L/2) + 1 / np.abs(nucSpace + L/2), (elecDim, nucDim)).flatten())
# plt.show()

t = 0
phiSave = np.zeros_like(phi)

# Measure time 0
elecRedProb, nucRedProb = f.getRedProbs(phi, elecDim, nucDim, elecDx, nucDx)
nucRedProbAll = [nucRedProb]
elecRedProbAll = [elecRedProb]
timeAll = [0]
nucStates = f.getBO(phi, elecEigenstates, numEigStates, nucDim, elecDim, elecDx)
nucPopAll = [np.sum(np.real(np.conj(nucStates) * nucStates), axis=0) * nucDx]
decDynAll = [f.getDecDyn(nucStates, numEigStates, nucDx)]

for i in range(iMax):
    if i % 10 == 0: print("\tCurrent state {:.1f} %".format(i / iMax * 100), end="\r", flush=True)
    phiNew = f.rk4(phi, dt, hamiltonian)
    phi = np.copy(phiNew)
    if (i + 1) % printEvery == 0:
        t = dt * (i + 1)
        elecRedProb, nucRedProb = f.getRedProbs(phi, elecDim, nucDim, elecDx, nucDx)
        nucRedProbAll.append(nucRedProb)
        elecRedProbAll.append(elecRedProb)
        nucStates = f.getBO(phi, elecEigenstates, numEigStates, nucDim, elecDim, elecDx)
        nucPopAll.append(np.sum(np.real(np.conj(nucStates) * nucStates), axis=0) * nucDx)
        decDynAll.append(f.getDecDyn(nucStates, numEigStates, nucDx))
        timeAll.append(t * AtomicToFs)

print("\tFinished Dynamics of {} femtoseconds.".format(tMax))
print("\tThe final norm is {}.".format(f.norm(phi, elecDx,nucDx)))

nucPopAll = np.array(nucPopAll)
decDynAll = np.array(decDynAll)

print("\n===========================================================\n")

###############################################################################

###############################################################################
################################## Animation ##################################
###############################################################################

print("=======================  ANIMATION  =======================\n")

figAnim = plt.figure(figsize=(6.4, 1.3 * 4.8))
figAnim.subplots_adjust(hspace=0.3) 

axAnimWF = figAnim.add_subplot(211)
axAnimWF.axvline(x = -L / 2, color='gray', label='ions')
axAnimWF.axvline(x = L / 2, color='gray')
elecAnim = axAnimWF.plot(elecSpace, elecRedProbAll[0], label=r"$\rho_e(r)$")
nucAnim = axAnimWF.plot(nucSpace, nucRedProbAll[0], label=r"$\rho_N(R)$")
axAnimWF.set_title("Time = {:.1f} fs".format(timeAll[0]))
axAnimWF.legend(loc='upper left', fontsize='x-small')
axAnimWF.set_xlim(-1.5 * L/2, 1.5 * L/2)
axAnimWF.set_ylim(0,np.max(nucRedProbAll)+0.05)
axAnimWF.set_xlabel('R and r (Bohr)')
axAnimWF.set_ylabel(r'PDF (Bohr$^{-1}$)')

axAnimPop = figAnim.add_subplot(223)
nucPlots = [axAnimPop.plot(timeAll[0], nucPopAll[0,i], label=r"State {}".format(i)) for i in range(numEigStates)]
axAnimPop.set_xlim((0,tMax))
axAnimPop.set_yscale('log')
axAnimPop.set_ylim((1e-5, 2))
axAnimPop.set_xlabel('Time (fs)')
axAnimPop.set_ylabel(r'$|\langle \varphi_i(r) | \Psi(r,R) \rangle|^2$')
axAnimPop.legend(loc='lower center', fontsize='x-small')

decDynLabels = []
for i in range(numEigStates):
    for j in range(i + 1, numEigStates):
        decDynLabels.append('{}{}'.format(i,j))
        
axAnimDec = figAnim.add_subplot(224)
decPlots = [axAnimDec.plot(timeAll[0], decDynAll[0,i], label=r"D$_{%s}$" %(decDynLabels[i])) for i in range(len(decDynLabels))]
axAnimDec.set_ylim((0, np.max(decDynAll)))
axAnimDec.set_xlim((0, tMax))
axAnimDec.legend(loc="upper right", fontsize='x-small')
axAnimDec.set_xlabel("Time (fs)")
axAnimDec.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

def animate(i, elecProb, nucProb, pops, decDynAll, time):
    elecAnim[0].set_ydata(elecProb[i])
    nucAnim[0].set_ydata(nucProb[i])
    axAnimWF.set_title("Time = {:.1f} fs".format(time[i]))

    [nucPlots[j][0].set_data(timeAll[:i], pops[:i,j]) for j in range(numEigStates)]

    [decPlots[j][0].set_data(timeAll[:i], decDynAll[:i,j]) for j in range(len(decDynLabels))]

animation = FuncAnimation(
    figAnim, 
    animate, 
    frames = range(1, int(iMax / printEvery) + 1), 
    fargs = (elecRedProbAll, nucRedProbAll, nucPopAll, decDynAll, timeAll),
    interval = 33.3
    )

animation.save(
    resultsDir + "/Results.mp4", 
    dpi = 300, 
    progress_callback = lambda j, n: print(f'\tSaving frame {j} of {n}', end="\r", flush=True)
    )

print("\tFinished Animation        ")

print("\n===========================================================\n")

###############################################################################

print("\tNORMAL TERMINATION\n")
