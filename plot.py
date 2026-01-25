import matplotlib.pyplot as plt

def plot_step(obs, name="Observable", BJ=0.0):
    plt.figure(figsize=(8,6))
    plt.plot(obs)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo step")
    plt.xlabel("Monte Carlo step")
    plt.ylabel(name)
    plt.legend([f"BJ = {BJ}"])
    #save figure    
    plt.savefig(f"{name.replace(' ', '_')}_BJ_{BJ}.png")
    plt.close()