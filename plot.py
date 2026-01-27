import matplotlib.pyplot as plt

#plot of an observable vs Monte Carlo steps

def plot_step(obs, name="Observable", BJ =0):
    plt.figure(figsize=(8,6))
    plt.plot(obs)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo step")
    plt.xlabel("Monte Carlo step")
    plt.ylabel(name)
    plt.legend([f"BJ = {BJ}"])
    #save figure    
    plt.savefig(f"{name.replace(' ', '_')}_B_{BJ}.png")
    plt.close()

#plot of 2 observables vs Monte Carlo steps
    
def plot_two_steps(obs1, obs2, name="Observable", name1="Observable 1", name2="Observable 2", BJ=0):
    plt.figure(figsize=(8,6))
    plt.plot(obs1, label=name1)
    plt.plot(obs2, label=name2)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo time at BJ={BJ}")
    plt.xlabel("t [Monte Carlo]")
    plt.ylabel(name)
    plt.legend([f"{name1})", f"{name2}"])
    #save figure    
    plt.savefig(f"{name}_{BJ}.png")
    plt.close()

#plot of an observable vs temperature
def plot_vs_BJ(BJ_s, obs_s, name="Observable"):
    plt.figure(figsize=(8,6))
    plt.plot(BJ_s, obs_s, marker='o')
    plt.grid()
    plt.title(f"{name} vs BJ")
    plt.xlabel("BJ")
    plt.ylabel(name)
    #save figure    
    plt.savefig(f"{name}_vs_BJ.png")
    plt.close()
    