import matplotlib.pyplot as plt

#plot of an observable vs Monte Carlo steps

def plot_step(obs, name="Observable", T=0):
    plt.figure(figsize=(8,6))
    plt.plot(obs)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo step")
    plt.xlabel("Monte Carlo step")
    plt.ylabel(name)
    plt.legend([f"T = {T}"])
    #save figure    
    plt.savefig(f"{name.replace(' ', '_')}_T_{T}.png")
    plt.close()

#plot of 2 observables vs Monte Carlo steps
    
def plot_two_steps(obs1, obs2, name1="Observable 1", name2="Observable 2", T=0):
    plt.figure(figsize=(8,6))
    plt.plot(obs1, label=name1)
    plt.plot(obs2, label=name2)
    plt.grid()
    plt.title(f"{name1} and {name2} vs Monte Carlo step")
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Value")
    plt.legend([f"{name1} (T={T})", f"{name2} (T={T})"])
    #save figure    
    plt.savefig(f"{name1.replace(' ', '_')}_and_{name2.replace(' ', '_')}_T_{T}.png")
    plt.close()

#plot of an observable vs temperature
    