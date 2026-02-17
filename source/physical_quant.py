import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def total_energy(spins):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = - spins * convolve(spins, kern, mode='wrap', cval=0) #periodic boundary conditions
    return arr.sum()/2

def abs_magnetization(spins):
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

def spatial_correlation(spins, r):
    mean_spin = np.mean(spins)
    horiz = np.mean(spins * np.roll(spins, r, axis=1))
    vert  = np.mean(spins * np.roll(spins, r, axis=0))
    return (horiz + vert) / 2 - mean_spin**2

def mean_spatial_correlation(configs, L, max_r=None):
    if max_r is None:
        max_r = L // 2
    
    C_r_mean = np.zeros(max_r)
    
    for config in configs:
        spins = config.reshape(L, L)
        for r in range(max_r):
            C_r_mean[r] += spatial_correlation(spins, r)
    
    C_r_mean /= len(configs)
    return np.arange(max_r), C_r_mean