import matplotlib.pyplot as plt


#plot of 2 observables vs Monte Carlo steps
def plot_two_steps(obs1, obs2, name="Observable", name1="Observable 1", name2="Observable 2", BJ=0):
    plt.figure(figsize=(8,6))
    plt.plot(obs1, label=name1)
    plt.plot(obs2, label=name2)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo time at BJ={BJ}")
    plt.xlabel("t [Monte Carlo]")
    plt.ylabel(name)
    plt.legend([f"{name1}", f"{name2}"])
    #save figure in figures folder
    plt.savefig(f"figures/{name}_{BJ}.png")
    plt.close()

def plot_vs_T_errors(T_s, obs_s, obs_err_s, name="Observable"):
    plt.figure(figsize=(8,6))
    plt.errorbar(T_s, obs_s, yerr=obs_err_s, fmt='o', capsize=5)
    plt.grid()
    plt.title(f"{name} vs T")
    plt.xlabel("T")
    plt.ylabel(name)
    #save figure in figures folder
    plt.savefig(f"figures/{name}_vs_T_errors.png")
    plt.close()