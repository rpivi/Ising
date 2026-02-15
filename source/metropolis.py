import numpy as np
import physical_quant as ph

def metropolis(spins, nsweep, sweeps_skip, BJ, return_configs=False):
    spins_arr = spins.copy()
    energy = ph.total_energy(spins_arr)
    L = spins_arr.shape[0]
    
    n_measurements = nsweep // sweeps_skip
    net_spins = np.zeros(n_measurements)
    abs_magnet = np.zeros(n_measurements)
    net_energies = np.zeros(n_measurements)
    configs = [] if return_configs else None
    
    measurement_counter = 0
    
    for sweep in range(nsweep):
        # One sweep = L*L proposed flips
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            
            spin_i = spins_arr[i, j]
            neighbors = (spins_arr[(i+1)%L, j] +
                        spins_arr[(i-1)%L, j] + 
                        spins_arr[i, (j+1)%L] +
                        spins_arr[i, (j-1)%L])
            
            dE = 2 * spin_i * neighbors
            
            if (dE <= 0) or (np.random.rand() < np.exp(-BJ*dE)):
                spins_arr[i, j] = -spin_i
                energy += dE
        
        # Measure after each sweep if appropriate
        if sweep % sweeps_skip == 0:
            net_spins[measurement_counter] = spins_arr.sum()
            abs_magnet[measurement_counter] = ph.abs_magnetization(spins_arr)
            net_energies[measurement_counter] = energy
            if return_configs:
                configs.append(spins_arr.copy())
            measurement_counter += 1
    
    if return_configs:
        return net_spins, abs_magnet, net_energies, spins_arr, configs
    else:
        return net_spins, abs_magnet, net_energies, spins_arr
