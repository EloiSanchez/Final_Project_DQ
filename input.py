# Control parameters
numEigStates = 3 # Number of eigenstates to find
L = 19. # Distance between the fixed ions
M = 1863. # proton mass

# Electron grid
elecDx = 0.6
elecFactSpace = 6

# Nucleus grid
nucDx = 0.03
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