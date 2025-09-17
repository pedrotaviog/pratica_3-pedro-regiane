# ============================================
# Questão 4 - Validação com Y2.npy
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
import control as ctl  
from atv3 import G_old, G_h, G_m, G_s, G_st

Y2 = np.load("data/Y2.npy", allow_pickle=True)

if Y2.ndim == 1:
    y2 = Y2
    t2 = np.arange(len(y2)) * 0.1
else:
    t2, y2 = Y2[:,0], Y2[:,1]

K2 = y2[-1]
y2_norm = y2 / K2

# Simular respostas dos modelos já identificados
t_sim = np.linspace(0, t2[-1], 500)
_, y_old = ctl.step_response(G_old, T=t_sim)
_, y_h = ctl.step_response(G_h, T=t_sim)
_, y_m = ctl.step_response(G_m, T=t_sim)
_, y_s = ctl.step_response(G_s, T=t_sim)
_, y_st = ctl.step_response(G_st, T=t_sim)

plt.figure(figsize=(10,6))
plt.plot(t2, y2_norm, 'k', label="Dados Y2 (normalizados)")
plt.plot(t_sim, y_old, label="Oldenbourg")
plt.plot(t_sim, y_h, label="Harriott")
plt.plot(t_sim, y_m, label="Meyer")
plt.plot(t_sim, y_s, label="Smith")
plt.plot(t_sim, y_st, label="Sten")
plt.xlabel("Tempo (s)")
plt.ylabel("y(t)")
plt.title("Validação dos modelos com Y2.npy")
plt.legend()
plt.grid()
plt.show()
