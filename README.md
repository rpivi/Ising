# Modello di Ising 2D

Simulazione e analisi del modello di Ising bidimensionale su reticolo quadrato, con teoria di campo medio, simulazioni Monte Carlo e Principal Component Analysis (PCA).

---

## Contenuto del progetto

Il progetto studia la transizione di fase ferromagnetica–paramagnetica attraverso tre approcci complementari:

- **Teoria di campo medio** — soluzione analitica approssimata e stima della temperatura critica
- **Simulazioni Monte Carlo** — algoritmo di Metropolis–Hastings per il campionamento della distribuzione di Boltzmann
- **PCA** — identificazione del parametro d'ordine e della transizione di fase tramite apprendimento non supervisionato.
  
- La relazione sul progetto è disponibile cliccando sul link: [Relazione(PDF)](report/main.pdf) 
---

## Il modello

Il sistema è descritto dall'Hamiltoniana:

$$E = -J \sum_{\langle i,j \rangle} s_i s_j$$

con spin $s_i \in \{+1, -1\}$ su reticolo quadrato $L \times L$, condizioni al contorno periodiche, e nel caso ferromagnetico ($J > 0$) in assenza di campo esterno.

---

## Metodi

### Teoria di campo medio

La temperatura critica stimata dalla teoria di campo medio è:

$$T_c^{\text{MF}} = \frac{zJ}{k_B} = \frac{4J}{k_B}$$

in accordo qualitativo con il valore esatto di Onsager:

$$T_c = \frac{2J}{k_B \ln(1+\sqrt{2})} \approx 2.269 \, J/k_B$$

### Monte Carlo (Metropolis–Hastings)

Le simulazioni sono state condotte su reticoli $L = 30$ e $L = 50$, calcolando le seguenti osservabili termodinamiche:

| Grandezza | Formula |
|---|---|
| Energia media | $\langle E \rangle$ |
| Magnetizzazione | $\langle \|M\| \rangle$ |
| Capacità termica| $C_V / k_b= \frac{\beta^2}{N}(\langle E^2 \rangle - \langle E \rangle^2)$ |
| Suscettività | $\chi = \frac{\beta}{N}(\langle M^2 \rangle - \langle M \rangle^2)$ |

### Stima della temperatura critica

| Dimensione $L$ | Osservabile | $T_c$ |
|---|---|---|
| 30 | $\chi$ | $2.3 \pm 0.1$ |
| 30 | $C_V$ | $2.3 \pm 0.2$ |
| 50 | $\chi$ | $2.27 \pm 0.06$ |
| 50 | $C_V$ | $2.3 \pm 0.1$ |

I risultati sono compatibili con il valore esatto di Onsager entro gli errori stimati.

### PCA

Applicando la PCA alle configurazioni Monte Carlo:

- La **prima componente principale** è correlata quasi perfettamente con la magnetizzazione ($r \approx 1$)
- La **seconda componente principale** riflette il comportamento della suscettività ($r \approx 0.7$)

La PCA riesce a identificare la transizione di fase e il parametro d'ordine senza alcuna conoscenza a priori della fisica del sistema.

---

## Implementazione

Il codice è scritto in **Python**.

## Risultati principali

- La transizione di fase è chiaramente visibile nel comportamento della magnetizzazione, del calore specifico e della suscettività
- Il picco di $\chi$ e $C_V$ in prossimità di $T_c$ è la firma caratteristica di una transizione di secondo ordine
- La PCA dimostra il potenziale degli strumenti di data science nell'esplorazione di sistemi fisici complessi

---

## Riferimenti

- P. W. Anderson, *More is Different*, Science, 1972
- E. Ising, *Contribution to the Theory of Ferromagnetism*, Zeitschrift für Physik, 1925
- L. Onsager, *Crystal Statistics*, Physical Review, 1944
- W. Hu, R. R. P. Singh, R. T. Scalettar, *Discovering phases, phase transitions and crossovers through unsupervised machine learning*, Physical Review E, 2017

---

## Autore

**Pivi Riccardo** — Marzo 2026
