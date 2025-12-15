import lattice as lat
import numpy as np
def local_energy(spins, coord, J):
    s = spins[coord]
    e = -J * s * (spins[coord[0], (coord[1] + 1) % spins.shape[1]] +
                  spins[coord[0],(coord[1] - 1) % spins.shape[1]] + 
                  spins[coord[0]+1 % spins.shape[0], coord[1]] + 
                  spins[(coord[0] - 1) % spins.shape[0], coord[1]])
    return e

def delta_energy(spins, coord, J):
    de = 2 * local_energy(spins, coord, J) #e_dopo_il_flip - e_prima_del_flip
    return de   

def total_energy(spins, J):
    E = 0
    N = spins.shape[0]
    for i in range(N):
        for j in range(N):
            E += -J * spins[i][j] * (spins[i][(j + 1) % N] + spins[(i + 1) % N][j]) 
    return E  # is it correct?