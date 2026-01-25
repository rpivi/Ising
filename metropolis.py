import numpy as np
import physical_quant as ph

# Metropolis algorithm step for the 2D Ising model
def metropolis_step(spins, BJ):
    L = spins.shape[0]
    i = np.random.randint(0,L)
    j = np.random.randint(0,L)
    coord = (i, j)
    dE = ph.delta_energy(spins,coord)
    dM = -2 * spins[i, j]

    if dE <= 0 or np.random.random() < np.exp(-dE * BJ): # kb = 1
        spins[i, j] *= -1 
        return dE, dM
    return 0, 0