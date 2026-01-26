import numpy as np
import physical_quant as ph

def metropolis(spins, times, BJ, energy):
    spins_arr = spins.copy()
    net_spins = np.zeros(times-1)
    net_energies = np.zeros(times-1)
    L = spins_arr.shape[0]

    for t in range(0,times-1):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        spin_i = spins_arr[i, j]
        spin_f = -spin_i

        # Calculate the change in energy
        spin_i = spins[i, j]
        neighbors = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] +spins[i, (j-1)%L] #periodic boundary conditions
        dE = 2 * spin_i * neighbors
        # Metropolis criterion
        if (dE <= 0 ) or (np.random.rand() < np.exp(-BJ * dE)):
            spins_arr[i, j] = spin_f
            energy += dE
        else:
            spins_arr[i, j] = spin_i
        if (t % 1000 == 0):
            print(f"Step {t}/{times-1} completed.")
            
        net_spins[t] = spins_arr.sum()
        net_energies[t] = energy
    return net_spins, net_energies