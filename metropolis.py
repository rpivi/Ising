import numpy as np
import physical_quant as ph

def metropolis(spins, nsweep, BJ):

    #settings 
    spins_arr = spins.copy()
    energy = ph.total_energy(spins_arr)
    L = spins_arr.shape[0]
    steps = nsweep * L * L
    net_spins = np.zeros(nsweep)
    net_energies = np.zeros(nsweep)
  
    for t in range(0, steps):   
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        spin_i = spins_arr[i, j]
        spin_f = -spin_i

        # Calculate the change in energy
        neighbors = (spins_arr[(i+1)%L, j] +
                     spins_arr[(i-1)%L, j] + 
                     spins_arr[i, (j+1)%L] +
                     spins_arr[i, (j-1)%L]) #periodic boundary conditions
        
        dE = 2 * spin_i * neighbors

        # Metropolis criterion
        if (dE <= 0 ) or (np.random.rand() < np.exp(-BJ * dE)):
            spins_arr[i, j] = spin_f
            energy += dE
        if ((t+1)% (L*L) == 0):
            s = (t+1)//(L*L) -1
            net_spins[s] = ph.magnetization(spins_arr)
            net_energies[s] = energy
    return net_spins, net_energies, spins_arr