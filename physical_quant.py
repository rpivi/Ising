import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def local_energy(spins, coord, J=1.0):
    L = spins.shape[0]
    i, j = coord
    s = spins[i, j]

    neighbors = (
        spins[(i+1) % L, j] +
        spins[(i-1) % L, j] +
        spins[i, (j+1) % L] +
        spins[i, (j-1) % L]
    )

    return -J * s * neighbors

def delta_energy(spins, coord, J):
    de = 2 * local_energy(spins, coord, J) #e_dopo_il_flip - e_prima_del_flip
    return de   

def total_energy(spins, J):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = -J * spins * convolve(spins, kern, mode='constant', cval=0)
    return arr.sum() #è doppio?
    
   
def magnetization(spins):
    return np.sum(spins)

