import numpy as np
from numba import njit
import physical_quant as ph

@njit(cache=True)
def _metropolis_core(spins_arr, nsweep, sweeps_skip, BJ):
    """
    Pure Numba kernel — no dicts, no Python objects.
    Returns arrays of (net_spins, abs_magnet, net_energies)
    and the final spin configuration.
    """
    L = spins_arr.shape[0]
    n_measurements = nsweep // sweeps_skip

    net_spins    = np.zeros(n_measurements)
    abs_magnet   = np.zeros(n_measurements)
    net_energies = np.zeros(n_measurements)

    # Precompute Boltzmann factors for the only two positive dE values
    # in 2D Ising: dE ∈ {4, 8}  (dE ≤ 0 always accepted)
    boltz4 = np.exp(-BJ * 4.0)
    boltz8 = np.exp(-BJ * 8.0)

    energy = ph.total_energy(spins_arr)   # must also be @njit
    measurement_counter = 0

    for sweep in range(nsweep):
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            spin_ij = spins_arr[i, j]

            neighbors = (spins_arr[(i + 1) % L,  j         ] +
                         spins_arr[(i - 1) % L,  j         ] +
                         spins_arr[ i,           (j + 1) % L] +
                         spins_arr[ i,           (j - 1) % L])

            dE = 2.0 * spin_ij * neighbors

            # Accept / reject
            if dE <= 0.0:
                spins_arr[i, j] = -spin_ij
                energy += dE
            else:
                # Inline lookup — avoids dict
                bf = boltz4 if dE == 4.0 else boltz8
                if np.random.random() < bf:
                    spins_arr[i, j] = -spin_ij
                    energy += dE

        # Measurement
        if sweep % sweeps_skip == 0:
            M = spins_arr.sum()
            net_spins   [measurement_counter] = M
            abs_magnet  [measurement_counter] = abs(M)
            net_energies[measurement_counter] = energy
            measurement_counter += 1

    return net_spins, abs_magnet, net_energies


def metropolis(spins, nsweep, sweeps_skip, BJ, return_configs=False):
    """ Python wrapper — handles optional config recording outside Numba.
    The hot loop runs entirely in compiled code."""
    spins_arr = spins.copy()

    if not return_configs:
        net_spins, abs_magnet, net_energies = _metropolis_core(
            spins_arr, nsweep, sweeps_skip, BJ
        )
        return net_spins, abs_magnet, net_energies, spins_arr

    # Config recording: run sweep-by-sweep, each call = sweeps_skip sweeps
    n_measurements = nsweep // sweeps_skip
    net_spins    = np.zeros(n_measurements)
    abs_magnet   = np.zeros(n_measurements)
    net_energies = np.zeros(n_measurements)
    configs      = []

    for k in range(n_measurements):
        ns, am, ne = _metropolis_core(spins_arr, sweeps_skip, sweeps_skip, BJ)
        net_spins   [k] = ns[-1]
        abs_magnet  [k] = am[-1]
        net_energies[k] = ne[-1]
        configs.append(spins_arr.copy())   # snapshot after each block

    return net_spins, abs_magnet, net_energies, spins_arr, configs

def get_therm_sweeps(T,L):
    delta=0.2
    if L == 30:
        nsweep_base=1000
        nsweep_critical=1500
    
    else:
        nsweep_base=2000
        nsweep_critical=3000

    # For temperatures close to the critical temperature T_c, we need more sweeps to ensure 
    #proper thermalization due to critical slowing down.
    T_c=2.269
    if abs(T - T_c) < delta:
        return nsweep_critical
    return nsweep_base
