import numpy as np
import matplotlib.pyplot as plt

# --- Função 1: Superamortecido (eq. 5)
def resposta_superamortecida(t, K=1, A=1, tau1=1.0, tau2=0.2):
    # Fórmula do slide:
    y = K*A*((1 - tau1*np.exp(-t/tau1) - tau2*np.exp(-t/tau2)) / (tau1 - tau2))
    return y

# --- Função 2: Criticamente amortecido (eq. 6)
def resposta_critica(t, K=1, A=1, tau=1.0):
    y = K*A*(1 - (1 + t/tau)*np.exp(-t/tau))
    return y

# --- Função 3: Subamortecido (eq. 7)
def resposta_subamortecida(t, K=1, A=1, xi=0.2, wn=5.0):
    wd = wn*np.sqrt(1 - xi**2)  # frequência amortecida
    y = K*A*(1 - np.exp(-xi*wn*t) *
             (np.cos(wd*t) + (xi/np.sqrt(1 - xi**2))*np.sin(wd*t)))
    return y

# Envelope para subamortecido (opcional)
def envelopes_subamortecido(t, K=1, A=1, xi=0.2, wn=5.0):
    env_sup = K*A*(1 + np.exp(-xi*wn*t))
    env_inf = K*A*(1 - np.exp(-xi*wn*t))
    return env_sup, env_inf

# --- Parâmetros de tempo
t = np.linspace(0, 5, 1000)

# --- Plots
plt.figure(figsize=(10,6))
plt.plot(t, resposta_superamortecida(t), label='Superamortecido')
plt.plot(t, resposta_critica(t), label='Criticamente Amortecido')
plt.plot(t, resposta_subamortecida(t), label='Subamortecido')

# Envelopes do subamortecido
env_sup, env_inf = envelopes_subamortecido(t)
plt.plot(t, env_sup, 'r--', alpha=0.5)
plt.plot(t, env_inf, 'r--', alpha=0.5)

plt.title('Respostas ao Degrau Unitário - Sistemas de 2ª Ordem')
plt.xlabel('Tempo (s)')
plt.ylabel('Saída y(t)')
plt.grid(True)
plt.legend()
plt.show()
