import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def total_energy(spins):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = - spins * convolve(spins, kern, mode='wrap', cval=0) #periodic boundary conditions
    return arr.sum()/2

def magnetization(spins):
    return abs(spins.sum())

def heat_capacity(energies, BJ, N):
    E2_mean = np.mean(np.array(energies)**2)
    E_mean = np.mean(energies)
    C = (BJ**2 / N) * (E2_mean - E_mean**2)
    return C

def susceptibility(magnetizations, BJ, N):
    M2_mean = np.mean(np.array(magnetizations)**2)
    M_mean = np.mean(magnetizations)
    chi = (BJ / N) * (M2_mean - M_mean**2)
    return chi