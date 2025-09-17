import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
import control as ctl  

# ============================================
# 1. Carregar dados experimentais
# ============================================
Y = np.load("data/Y.npy", allow_pickle=True)

if Y.ndim == 1:
    y = Y
    t = np.arange(len(y)) * 0.1  # passo de tempo estimado (ajuste se necessário)
else:
    t, y = Y[:,0], Y[:,1]

# ============================================
# 2. Normalização e parâmetros básicos
# ============================================
K = y[-1]
y_norm = y / K
print(f"Ganho estacionário K = {K:.4f}")

def tempo_para_valor(y, t, frac):
    alvo = frac * y[-1]
    idx = np.where(y >= alvo)[0][0]
    return t[idx]

TA = tempo_para_valor(y_norm, t, 0.1)
T90 = tempo_para_valor(y_norm, t, 0.9)
TC = T90 - TA
t20 = tempo_para_valor(y_norm, t, 0.2)
t60 = tempo_para_valor(y_norm, t, 0.6)
t73 = tempo_para_valor(y_norm, t, 0.73)

# derivada para ponto de inflexão
dy = savgol_filter(y_norm, 11, 3, deriv=1, delta=t[1]-t[0])
TI = t[np.argmax(dy)]

print(f"TA={TA:.2f}, TC={TC:.2f}, t20={t20:.2f}, t60={t60:.2f}, t73={t73:.2f}, TI={TI:.2f}")

# ============================================
# 3. Métodos de identificação
# ============================================

# Oldenbourg
def metodo_oldenbourg(TA, TC):
    r = TC / TA
    def eq(x): return (1+x)*x/(1-x) - r
    x = fsolve(eq, 0.5)[0]
    tau2 = TC / (1 + x)
    tau1 = x * tau2
    return tau1, tau2

tau1_old, tau2_old = metodo_oldenbourg(TA, TC)

# Harriott
def metodo_harriott(t73):
    tau_sum = t73 / 1.3
    tau1 = tau_sum / 2
    tau2 = tau_sum - tau1
    return tau1, tau2

tau1_h, tau2_h = metodo_harriott(t73)

# Meyer
def metodo_meyer(t20, t60):
    tau = t60 - t20
    wn = 1 / tau
    xi = 0.5  # chute aproximado (na prática usa gráfico 20/60)
    return wn, xi

wn_m, xi_m = metodo_meyer(t20, t60)

# Smith – simplificado
def metodo_smith(TA, TC, TB=1.0):
    TF = TA - TC
    tau1, tau2 = metodo_oldenbourg(TA, TC)
    theta = TF  # pode sair negativo
    return tau1, tau2, theta

tau1_s, tau2_s, theta_s = metodo_smith(TA, TC)

# Sten – atraso via fórmula
def metodo_sten(TA, TC, TI, TB=1.0):
    tau1, tau2 = metodo_oldenbourg(TA, TC)
    theta = TI + TC - TA - TB
    return tau1, tau2, theta

tau1_st, tau2_st, theta_st = metodo_sten(TA, TC, TI)

print(f"Oldenbourg: tau1={tau1_old:.2f}, tau2={tau2_old:.2f}")
print(f"Harriott:   tau1={tau1_h:.2f}, tau2={tau2_h:.2f}")
print(f"Meyer:      wn={wn_m:.2f}, xi={xi_m:.2f}")
print(f"Smith:      tau1={tau1_s:.2f}, tau2={tau2_s:.2f}, theta={theta_s:.2f}")
print(f"Sten:       tau1={tau1_st:.2f}, tau2={tau2_st:.2f}, theta={theta_st:.2f}")

# ============================================
# 4. Construir modelos no control
# ============================================

# Oldenbourg
G_old = ctl.TransferFunction([K], [tau1_old*tau2_old, tau1_old+tau2_old, 1])

# Harriott
G_h = ctl.TransferFunction([K], [tau1_h*tau2_h, tau1_h+tau2_h, 1])

# Meyer
G_m = ctl.TransferFunction([wn_m**2], [1, 2*xi_m*wn_m, wn_m**2])

# Smith
theta_s = max(theta_s, 0)  # corrigir atraso negativo
num_pade, den_pade = ctl.pade(theta_s, 1)
G_s = ctl.TransferFunction([K], [tau1_s*tau2_s, tau1_s+tau2_s, 1]) * ctl.TransferFunction(num_pade, den_pade)

# Sten
theta_st = max(theta_st, 0)  # corrigir atraso negativo
num_pade, den_pade = ctl.pade(theta_st, 1)
G_st = ctl.TransferFunction([K], [tau1_st*tau2_st, tau1_st+tau2_st, 1]) * ctl.TransferFunction(num_pade, den_pade)

# ============================================
# 5. Simulação e comparação
# ============================================
t_sim = np.linspace(0, t[-1], 500)
_, y_old = ctl.step_response(G_old, T=t_sim)
_, y_h = ctl.step_response(G_h, T=t_sim)
_, y_m = ctl.step_response(G_m, T=t_sim)
_, y_s = ctl.step_response(G_s, T=t_sim)
_, y_st = ctl.step_response(G_st, T=t_sim)

plt.figure(figsize=(10,6))
plt.plot(t, y_norm, 'k', label="Dados reais (normalizados)")
plt.plot(t_sim, y_old, label="Oldenbourg")
plt.plot(t_sim, y_h, label="Harriott")
plt.plot(t_sim, y_m, label="Meyer")
plt.plot(t_sim, y_s, label="Smith")
plt.plot(t_sim, y_st, label="Sten")
plt.xlabel("Tempo (s)")
plt.ylabel("y(t)")
plt.title("Comparação entre dados reais e modelos identificados")
plt.legend()
plt.grid()
plt.show()