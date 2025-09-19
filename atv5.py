import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import savgol_filter, TransferFunction, step

# =========================================================
# 1) Leitura dos dados (formato: "Angle: xx, Duty_cycle: yy%, Timer: zzms")
# =========================================================
def load_fanplate_txt(path):
    angles, duty, time = [], [], []
    with open(path, 'r') as f:
        for line in f:
            m = re.findall(r"Angle:\s*([-\d.]+),\s*Duty_cycle:\s*(\d+)%.*Timer:\s*(\d+)ms", line)
            if m:
                a, d, t = m[0]
                angles.append(float(a))
                duty.append(int(d))
                time.append(int(t))
    t = np.array(time, dtype=float)/1000.0  # s
    y = np.array(angles, dtype=float)
    u = np.array(duty, dtype=float)
    return t, y, u

def normalize_to_01(y):
    # normaliza no intervalo [0,1] tomando início e final do ensaio
    y0 = y[0]
    yf = y[-1]
    den = (yf - y0) if abs(yf - y0) > 1e-9 else 1.0
    return (y - y0) / den

# =========================================================
# 2) Extração de tempos característicos a partir de y_norm(t)
#    TA: instante de 10%; TC: T90 - T10; t20,t60,t73; TI: ponto de inflexão;
#    TB: interseção da tangente em TI com a linha y(0)
# =========================================================
def time_at_fraction(t, y_norm, frac):
    # retorna o primeiro instante em que y_norm >= frac
    idx = np.where(y_norm >= frac)[0]
    return t[idx[0]] if idx.size else t[-1]

def characteristic_times(t, y_norm):
    # garantias básicas
    t = t - t[0]
    # 10% e 90%
    T10 = time_at_fraction(t, y_norm, 0.10)
    T90 = time_at_fraction(t, y_norm, 0.90)
    TA  = T10
    TC  = T90 - T10
    # 20%, 60%, 73%
    t20 = time_at_fraction(t, y_norm, 0.20)
    t60 = time_at_fraction(t, y_norm, 0.60)
    t73 = time_at_fraction(t, y_norm, 0.73)
    # ponto de inflexão (máx da derivada suave)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    win = max(5, min(51, 1 + 2*(len(t)//20)))  # janela ímpar
    if win % 2 == 0: win += 1
    dy = savgol_filter(y_norm, win, 3, deriv=1, delta=dt)
    i_TI = int(np.argmax(dy))
    TI   = t[i_TI]
    # tangente em TI: y_tan(t) = m*(t - TI) + y(TI)
    m    = dy[i_TI]
    y_TI = y_norm[i_TI]
    # TB: quando a tangente cruza o nível inicial y(0) (≈ 0 após normalização)
    y0   = y_norm[0]
    if abs(m) < 1e-12:
        TB = TI  # degenera; melhor palpite
    else:
        t_int = TI + (y0 - y_TI)/m
        TB = max(0.0, t_int)  # tempo relativo ao início
    # 'a': valor real da curva no instante em que a tangente passa por y(0)
    # (interpolar y_norm em t_int)
    def interp(x, xt, yt):
        return np.interp(x, xt, yt)
    a = float(interp(TB, t, y_norm))  # y(t) no ponto onde a tangente cruza y0
    return TA, TC, t20, t60, t73, TI, TB, a

# =========================================================
# 3) Métodos clássicos
#    (fórmulas conforme os slides)
# =========================================================

# 3.1 Oldenbourg & Sartorius (eqs. 16 e 17)
def metodo_oldenbourg(TA, TC):
    # resolve (TC/TA) = ((1+x)*x/(1-x)) com x = tau1/tau2
    r = TC/TA
    # resolver para x numericamente (busca simples em (0,1))
    xs = np.linspace(1e-4, 0.999, 20000)
    f  = ((1+xs)*xs/(1-xs)) - r
    i  = np.argmin(np.abs(f))
    x  = xs[i]
    tau2 = TC/(1 + x)  # (17)
    tau1 = x * tau2
    return max(tau1, 1e-6), max(tau2, 1e-6)

# 3.2 Smith (eqs. 19–24)
def metodo_smith(TA, TC, TB, a):
    # TF = TA - TC (21)
    TF = TA - TC
    # τ1 pelas eqs. (22) ou (23), conforme 'a'
    e = np.e
    if a <= 0.005:
        tau1 = TB*(1 + 10*a + (e - 1.0)*(30*a)**2)   # (22)
    else:
        denom = 1.0 + 0.086 + ((0.0015/(0.032 - a))**(-1))  # ~ 1.086 + (0.032-a)/0.0015
        tau1 = (TB + TF)*(1 - 200*(0.032 - a)/denom)        # (23)
    tau2 = TC - tau1                                         # (24)
    # Atraso (não aparece explícito nos slides de Smith): adotamos θ = TF
    theta = max(TF, 0.0)
    # sanear
    tau1 = max(float(tau1), 1e-6)
    tau2 = max(float(tau2), 1e-6)
    return tau1, tau2, float(theta)

# 3.3 Sten (eq. 26)
def metodo_sten(TA, TC, TI, TB):
    tau1, tau2 = metodo_oldenbourg(TA, TC)  # segue Oldenbourg para τ1,τ2
    theta = TI + TC - TA - TB               # (26)
    return tau1, tau2, max(float(theta), 0.0)

# 3.4 Harriott (eq. 28) — sem o gráfico, usamos divisão simples
def metodo_harriott(t73):
    tau_sum = t73/1.3  # (28)
    tau1 = 0.5*tau_sum
    tau2 = tau_sum - tau1
    return max(tau1, 1e-6), max(tau2, 1e-6)

# 3.5 Meyer — precisa do gráfico t20/t60→(ξ, t60/τ); aqui fazemos aproximação
def metodo_meyer(t20, t60, theta=0.0):
    # τ ~ (t60 - t20); ωn = 1/τ (34). ξ obteria-se do gráfico; aqui usamos aproximação neutra.
    tau = max(t60 - t20, 1e-6)
    wn  = 1.0/tau
    xi  = 0.5  # ajuste aproximado sem consultar a curva do slide
    return wn, xi

# =========================================================
# 4) Simulação dos modelos (com ou sem atraso)
# =========================================================
def step_two_real_poles(K, tau1, tau2, t):
    num = [K]
    den = [tau1*tau2, tau1 + tau2, 1.0]
    sys = TransferFunction(num, den)
    tout, y = step(sys, T=t)
    return y

def step_second_order_std(wn, xi, t, K=1.0):
    num = [K*wn**2]
    den = [1.0, 2*xi*wn, wn**2]
    sys = TransferFunction(num, den)
    tout, y = step(sys, T=t)
    return y

def apply_deadtime_shift(y, t, theta):
    if theta <= 0:
        return y
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.01
    n  = int(np.round(theta/dt))
    if n <= 0: 
        return y
    y_shift = np.concatenate([np.zeros(n), y])[:len(y)]
    return y_shift

# =========================================================
# 5) Execução principal
# =========================================================
if __name__ == "__main__":
    # --- Carregar identificação e validação
    t1, y1, u1 = load_fanplate_txt("data/Dados_degrau_FanPlate.txt")
    t2, y2, u2 = load_fanplate_txt("data/Dados_degrau_FanPlate_2.txt")

    # Normalizar saídas
    y1n = normalize_to_01(y1)
    y2n = normalize_to_01(y2)
    t1  = t1 - t1[0]
    t2  = t2 - t2[0]

    # --- Extrair tempos característicos no dataset de identificação
    TA, TC, t20, t60, t73, TI, TB, a = characteristic_times(t1, y1n)
    print(f"Característicos: TA={TA:.2f}, TC={TC:.2f}, t20={t20:.2f}, t60={t60:.2f}, t73={t73:.2f}, TI={TI:.2f}, TB={TB:.2f}, a={a:.4f}")

    # --- Métodos
    # Oldenbourg
    tau1_o, tau2_o = metodo_oldenbourg(TA, TC)

    # Smith
    tau1_s, tau2_s, theta_s = metodo_smith(TA, TC, TB, a)

    # Sten
    tau1_st, tau2_st, theta_st = metodo_sten(TA, TC, TI, TB)

    # Harriott
    tau1_h, tau2_h = metodo_harriott(t73)

    # Meyer
    wn_m, xi_m = metodo_meyer(t20, t60)

    print(f"Oldenbourg: tau1={tau1_o:.3f}, tau2={tau2_o:.3f}")
    print(f"Smith:      tau1={tau1_s:.3f}, tau2={tau2_s:.3f}, theta={theta_s:.3f}")
    print(f"Sten:       tau1={tau1_st:.3f}, tau2={tau2_st:.3f}, theta={theta_st:.3f}")
    print(f"Harriott:   tau1={tau1_h:.3f}, tau2={tau2_h:.3f}")
    print(f"Meyer:      wn={wn_m:.3f}, xi={xi_m:.3f}")

    # --- Simulações (identificação)
    K = 1.0  # saídas normalizadas
    y_old_id = step_two_real_poles(K, tau1_o,  tau2_o,  t1)
    y_har_id = step_two_real_poles(K, tau1_h,  tau2_h,  t1)
    y_smi_id = apply_deadtime_shift(step_two_real_poles(K, tau1_s,  tau2_s,  t1), t1, theta_s)
    y_ste_id = apply_deadtime_shift(step_two_real_poles(K, tau1_st, tau2_st, t1), t1, theta_st)
    y_mey_id = step_second_order_std(wn_m, xi_m, t1, K=K)

    # --- Simulações (validação)
    y_old_va = step_two_real_poles(K, tau1_o,  tau2_o,  t2)
    y_har_va = step_two_real_poles(K, tau1_h,  tau2_h,  t2)
    y_smi_va = apply_deadtime_shift(step_two_real_poles(K, tau1_s,  tau2_s,  t2), t2, theta_s)
    y_ste_va = apply_deadtime_shift(step_two_real_poles(K, tau1_st, tau2_st, t2), t2, theta_st)
    y_mey_va = step_second_order_std(wn_m, xi_m, t2, K=K)

    # --- Plots (identificação)
    plt.figure(figsize=(11,6))
    plt.plot(t1, y1n, 'k', lw=2, label='Dados (id)')
    plt.plot(t1, y_old_id, '--', label='Oldenbourg')
    plt.plot(t1, y_har_id, '--', label='Harriott')
    plt.plot(t1, y_smi_id, '--', label='Smith')
    plt.plot(t1, y_ste_id, '--', label='Sten')
    plt.plot(t1, y_mey_id, '--', label='Meyer')
    plt.xlabel('Tempo [s]'); plt.ylabel('Saída normalizada'); plt.grid(True)
    plt.title('Fan-Plate — identificação (Q5)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plots (validação)
    plt.figure(figsize=(11,6))
    plt.plot(t2, y2n, 'k', lw=2, label='Dados (val)')
    plt.plot(t2, y_old_va, '--', label='Oldenbourg')
    plt.plot(t2, y_har_va, '--', label='Harriott')
    plt.plot(t2, y_smi_va, '--', label='Smith')
    plt.plot(t2, y_ste_va, '--', label='Sten')
    plt.plot(t2, y_mey_va, '--', label='Meyer')
    plt.xlabel('Tempo [s]'); plt.ylabel('Saída normalizada'); plt.grid(True)
    plt.title('Fan-Plate — validação (Q5)')
    plt.legend()
    plt.tight_layout()
    plt.show()
