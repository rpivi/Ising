import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def total_energy(spins):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = - spins * convolve(spins, kern, mode='wrap', cval=0) #periodic boundary conditions
    return arr.sum()/2