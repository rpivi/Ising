import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def local_energy(spins, coord):
    L = spins.shape[0]
    i, j = coord
    s = spins[i, j]

    neighbors = (
        spins[(i+1) % L, j] +
        spins[(i-1) % L, j] +
        spins[i, (j+1) % L] +
        spins[i, (j-1) % L]
    )

    return -s * neighbors

def delta_energy(spins, coord):
    de = 2 * local_energy(spins, coord) # e_after - e_before = 2 * e_before
    return de   

def total_energy(spins):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = - spins * convolve(spins, kern, mode='wrap', cval=0)
    return arr.sum()/2
    
def magnetization(spins):
    return np.sum(spins)
