# Control parameters
numEigStates = 3 # Number of eigenstates to find
L = 19. # Distance between the fixed ions
M = 1863. # proton mass

# Electron grid
elecDx = 0.6
elecFactSpace = 6

# Nucleus grid
nucDx = 0.04
nucFactSpace = 0.99

# Interaction parameters
leftR = 4.4
rightR = 3.1
elecNucR = 7.

# Dynamic parameters
dt = 0.1  # Atomic units
printEvery = 20
tMax = 50  # Femtoseconds
AtomicToFs = 2.4189e-2

# Initial Gaussian wavepacket conditions
rn0 = -7.
sigma = 1. / (2.85)**(0.5)

# Initial Electronic Eigenstate
# = 0 (GS); = 1 (1st excited), = 2 (2nd excited), etc.
initEigenstate = 1

# Directory name to save the results
resultsDir = 'defaultResults'