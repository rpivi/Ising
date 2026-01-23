import matplotlib.pyplot as plt

def plot_step(obs, name="Observable"):
    plt.plot(obs)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo step")
    plt.xlabel("Monte Carlo step")
    plt.ylabel(name)
    plt.show()
    #save figure    
    plt.savefig(f"{name.replace('/', '_per_')}_vs_MC_step.png")