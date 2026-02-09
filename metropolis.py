import numpy as np
import physical_quant as ph

def metropolis(spins, nsweep,sweeps_skip ,BJ, return_configs=False):
    spins_arr = spins.copy()
    energy = ph.total_energy(spins_arr)
    L = spins_arr.shape[0]
    steps = nsweep * L * L
    n_measurements = nsweep // sweeps_skip
    net_spins = np.zeros(n_measurements)
    net_energies = np.zeros(n_measurements)
    configs = []

    for t in range(steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        spin_i = spins_arr[i, j]
        spin_f = -spin_i

        neighbors = (spins_arr[(i+1)%L, j] +
                     spins_arr[(i-1)%L, j] + 
                     spins_arr[i, (j+1)%L] +
                     spins_arr[i, (j-1)%L])
        dE = 2 * spin_i * neighbors

        if (dE <= 0) or (np.random.rand() < np.exp(-BJ*dE)):
            spins_arr[i, j] = spin_f
            energy += dE

        if (t+1) % (L*L) == 0:
            sweep_idx = (t+1)//(L*L) - 1
        if sweep_idx % sweeps_skip == 0:
            save_idx = sweep_idx // sweeps_skip
            net_spins[save_idx] = ph.abs_magnetization(spins_arr)
            net_energies[save_idx] = energy
            if return_configs:
                configs.append(spins_arr.copy())

    if return_configs:
        return net_spins, net_energies, spins_arr, configs
    else:
        return net_spins, net_energies, spins_arr
