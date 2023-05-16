# Final_Project_DQ

The code of this github page performs a quantum dynamics simulation using the Shin-Metiu model (dynamics of a proton and an electron under the potential created by two fixed ions) and outputs different observables from the simulation. The results include:

- Plot of the BOPES with the second order non-adiabatic couplings (NACs)
- Plot of the adiabatic electronic States
- Plot of the potential energy surfaces in the full-space
- mp4 file containing the time evolution of the reduced probability densities of proton and electrons, of the adiabatic populations and of the decoherence dynamics coefficients

The results aim to reproduce the ones from: Albareda, Guillermo, et al. "Correlated electron-nuclear dynamics with conditional wave functions." Physical review letters 113.8 (2014): 083003.

## Main files
- input.py : input parameters
- main.py : main program 
- Funcs.py : rellevant functions such as normalization or rk4 integrator
- operators.py : important operators such as the laplacian or the soft-core Coulomb interaction term

## Usage
The desired input parameters can be modified in the `input.py` file. To run the program one must type in the terminal: `python3 main.py`. After performing a simulation the results will be stored in the directory specified in the input file. 

## Results for Ground State

**Animation**

https://github.com/EloiSanchez/Final_Project_DQ/assets/79266117/ef7b6517-7b3f-47c0-b83d-9d173d0e1d4b

**Eigenstates**

![Eigenstates for groundstate](https://github.com/EloiSanchez/Final_Project_DQ/blob/main/groundState/EigenStates.png)

**Eigenvalues**

![Eigenvalues for groundstate](https://github.com/EloiSanchez/Final_Project_DQ/blob/main/groundState/EigenVals.png)

## Contributors

| Laia Barjuan                                                                   |Eloi Sanchez                                                                   |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| ![laiabarjuan](https://avatars.githubusercontent.com/u/79266111 "laiabarjuan") | ![EloiSanchez](https://avatars.githubusercontent.com/u/79266117 "EloiSanchez") |
| [laiabarjuan](https://github.com/laiabarjuan)                                  | [EloiSanchez](https://github.com/EloiSanchez)                                  |



