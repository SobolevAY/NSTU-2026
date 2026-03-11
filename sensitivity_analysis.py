# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# =========================================================
# 0. ПОДГОТОВКА ПАПКИ ДЛЯ ГРАФИКОВ
# =========================================================
PAPKA_GRAFIKOV = "figures"
os.makedirs(PAPKA_GRAFIKOV, exist_ok=True)

# =========================================================
# 1. БАЗОВАЯ МОДЕЛЬ СРЕДЫ
# =========================================================
# rho_b  : удельное сопротивление скважины
# rho_iz : удельное сопротивление зоны проникновения
# rho_f  : удельное сопротивление пласта
# r_b    : радиус скважины
# r_iz   : радиус зоны проникновения

rho_b = 1.0
rho_iz = 30.0
rho_f = 10.0
r_b = 0.1
r_iz = 0.5

sigma_perehoda = 0.5  # ширина сглаженного перехода

model = (rho_b, rho_iz, rho_f, r_b, r_iz)

# =========================================================
# 2. ПРЯМАЯ ЗАДАЧА: СГЛАЖЕННАЯ МОДЕЛЬ
# =========================================================
def forward_smooth(model, L, sigma=0.5):
    """
    Вычисляет кажущееся удельное сопротивление,
    измеряемое зондом длины L, для радиальной трехзонной модели
    со сглаженными переходами.
    """
    rho_b_local, rho_iz_local, rho_f_local, r_b_local, r_iz_local = model

    # защита от log(0)
    L = max(L, 1e-6)
    r_b_local = max(r_b_local, 1e-6)
    r_iz_local = max(r_iz_local, 1e-6)

    logL = np.log(L)
    log_rb = np.log(r_b_local)
    log_riz = np.log(r_iz_local)

    w_b = 0.5 * (1.0 - erf((logL - log_rb) / (sigma * np.sqrt(2.0))))
    w_iz = 0.5 * (
        erf((logL - log_rb) / (sigma * np.sqrt(2.0))) -
        erf((logL - log_riz) / (sigma * np.sqrt(2.0)))
    )
    w_f = 0.5 * (1.0 + erf((logL - log_riz) / (sigma * np.sqrt(2.0))))

    sigma_eff = w_b / rho_b_local + w_iz / rho_iz_local + w_f / rho_f_local

    if sigma_eff <= 0:
        return np.inf

    return 1.0 / sigma_eff


# Алиас, как в README
forward = forward_smooth

# =========================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def raschet_signala_pri_izmenenii_parametra(
    znacheniya_parametra, imya_parametra, bazovaya_model, L, sigma=0.5
):
    """
    Вычисляет сигнал при изменении одного параметра модели.
    imya_parametra: 'rho_iz' или 'r_iz'
    """
    rho_b_local, rho_iz_local, rho_f_local, r_b_local, r_iz_local = bazovaya_model
    signaly = []

    for znachenie in znacheniya_parametra:
        if imya_parametra == "rho_iz":
            tek_model = (rho_b_local, float(znachenie), rho_f_local, r_b_local, r_iz_local)
        elif imya_parametra == "r_iz":
            tek_model = (rho_b_local, rho_iz_local, rho_f_local, r_b_local, float(znachenie))
        else:
            raise ValueError("imya_parametra должно быть 'rho_iz' или 'r_iz'")

        signal = forward(tek_model, L, sigma=sigma)
        signaly.append(signal)

    return np.array(signaly, dtype=float)


def absolyutnaya_chuvstvitelnost(znacheniya_parametra, signaly):
    """
    Абсолютная чувствительность: dS/dp
    """
    return np.gradient(signaly, znacheniya_parametra)


def otnositelnaya_chuvstvitelnost(znacheniya_parametra, signaly):
    """
    Относительная чувствительность:
    d(log S)/d(log p) = (p/S) * dS/dp
    """
    dSdp = np.gradient(signaly, znacheniya_parametra)
    eps = 1e-12
    return (znacheniya_parametra / (signaly + eps)) * dSdp


def pechat_max_chuvstvitelnosti(znacheniya_parametra, chuvstvitelnost, zagolovok):
    indeks = np.argmax(np.abs(chuvstvitelnost))
    print(f"{zagolovok}:")
    print(f"  максимум |чувствительности| при p = {znacheniya_parametra[indeks]:.6f}")
    print(f"  значение чувствительности = {chuvstvitelnost[indeks]:.6f}")
    print()


def sohranit_i_pokazat_grafik(imya_fayla):
    """
    Сохраняет текущий график в папку figures и показывает его.
    """
    put_k_faylu = os.path.join(PAPKA_GRAFIKOV, imya_fayla)
    plt.tight_layout()
    plt.savefig(put_k_faylu, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# 4. ИССЛЕДОВАНИЕ ЧУВСТВИТЕЛЬНОСТИ К rho_iz
# =========================================================
L_fix = 0.5  # фиксированная длина зонда

znacheniya_rho_iz = np.linspace(1.0, 100.0, 300)

signaly_rho = raschet_signala_pri_izmenenii_parametra(
    znacheniya_rho_iz, "rho_iz", model, L_fix, sigma=sigma_perehoda
)

abs_chuvstv_rho = absolyutnaya_chuvstvitelnost(znacheniya_rho_iz, signaly_rho)
otn_chuvstv_rho = otnositelnaya_chuvstvitelnost(znacheniya_rho_iz, signaly_rho)

print("=== Исследование чувствительности к rho_iz ===")
pechat_max_chuvstvitelnosti(
    znacheniya_rho_iz, abs_chuvstv_rho, "Абсолютная чувствительность"
)
pechat_max_chuvstvitelnosti(
    znacheniya_rho_iz, otn_chuvstv_rho, "Относительная чувствительность"
)

plt.figure(figsize=(10, 6))
plt.plot(znacheniya_rho_iz, signaly_rho, linewidth=2)
plt.xlabel("Удельное сопротивление зоны проникновения ρ_из (Ом·м)")
plt.ylabel("Сигнал / кажущееся сопротивление (Ом·м)")
plt.title(f"Сигнал зонда как функция ρ_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
sohranit_i_pokazat_grafik("signal_vs_rho_iz.png")

plt.figure(figsize=(10, 6))
plt.plot(
    znacheniya_rho_iz,
    abs_chuvstv_rho,
    linewidth=2,
    label="Абсолютная чувствительность dS/dρ_из"
)
plt.xlabel("ρ_из (Ом·м)")
plt.ylabel("dS/dρ_из")
plt.title(f"Абсолютная чувствительность к ρ_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("absolute_sensitivity_rho_iz.png")

plt.figure(figsize=(10, 6))
plt.plot(
    znacheniya_rho_iz,
    otn_chuvstv_rho,
    linewidth=2,
    label="Относительная чувствительность d(logS)/d(logρ_из)"
)
plt.xlabel("ρ_из (Ом·м)")
plt.ylabel("Относительная чувствительность")
plt.title(f"Относительная чувствительность к ρ_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("relative_sensitivity_rho_iz.png")

# =========================================================
# 5. ИССЛЕДОВАНИЕ ЧУВСТВИТЕЛЬНОСТИ К r_iz
# =========================================================
znacheniya_r_iz = np.linspace(0.05, 2.0, 300)

signaly_r = raschet_signala_pri_izmenenii_parametra(
    znacheniya_r_iz, "r_iz", model, L_fix, sigma=sigma_perehoda
)

abs_chuvstv_r = absolyutnaya_chuvstvitelnost(znacheniya_r_iz, signaly_r)
otn_chuvstv_r = otnositelnaya_chuvstvitelnost(znacheniya_r_iz, signaly_r)

print("=== Исследование чувствительности к r_iz ===")
pechat_max_chuvstvitelnosti(
    znacheniya_r_iz, abs_chuvstv_r, "Абсолютная чувствительность"
)
pechat_max_chuvstvitelnosti(
    znacheniya_r_iz, otn_chuvstv_r, "Относительная чувствительность"
)

plt.figure(figsize=(10, 6))
plt.plot(znacheniya_r_iz, signaly_r, linewidth=2)
plt.xlabel("Радиус зоны проникновения r_из (м)")
plt.ylabel("Сигнал / кажущееся сопротивление (Ом·м)")
plt.title(f"Сигнал зонда как функция r_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
sohranit_i_pokazat_grafik("signal_vs_r_iz.png")

plt.figure(figsize=(10, 6))
plt.plot(
    znacheniya_r_iz,
    abs_chuvstv_r,
    linewidth=2,
    label="Абсолютная чувствительность dS/dr_из"
)
plt.xlabel("r_из (м)")
plt.ylabel("dS/dr_из")
plt.title(f"Абсолютная чувствительность к r_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("absolute_sensitivity_r_iz.png")

plt.figure(figsize=(10, 6))
plt.plot(
    znacheniya_r_iz,
    otn_chuvstv_r,
    linewidth=2,
    label="Относительная чувствительность d(logS)/d(logr_из)"
)
plt.xlabel("r_из (м)")
plt.ylabel("Относительная чувствительность")
plt.title(f"Относительная чувствительность к r_из (L = {L_fix} м)")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("relative_sensitivity_r_iz.png")

# =========================================================
# 6. СРАВНЕНИЕ ЗОНДОВ РАЗНОЙ ДЛИНЫ
# =========================================================
dliny_zondov = [0.2, 0.5, 1.0, 1.5]

plt.figure(figsize=(10, 6))
for L in dliny_zondov:
    signaly = raschet_signala_pri_izmenenii_parametra(
        znacheniya_rho_iz, "rho_iz", model, L, sigma=sigma_perehoda
    )
    otn_chuvstv = otnositelnaya_chuvstvitelnost(znacheniya_rho_iz, signaly)
    plt.plot(znacheniya_rho_iz, otn_chuvstv, linewidth=2, label=f"L = {L} м")

plt.xlabel("ρ_из (Ом·м)")
plt.ylabel("Относительная чувствительность")
plt.title("Относительная чувствительность к ρ_из для зондов разной длины")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("multi_probe_relative_sensitivity_rho_iz.png")

plt.figure(figsize=(10, 6))
for L in dliny_zondov:
    signaly = raschet_signala_pri_izmenenii_parametra(
        znacheniya_r_iz, "r_iz", model, L, sigma=sigma_perehoda
    )
    otn_chuvstv = otnositelnaya_chuvstvitelnost(znacheniya_r_iz, signaly)
    plt.plot(znacheniya_r_iz, otn_chuvstv, linewidth=2, label=f"L = {L} м")

plt.xlabel("r_из (м)")
plt.ylabel("Относительная чувствительность")
plt.title("Относительная чувствительность к r_из для зондов разной длины")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("multi_probe_relative_sensitivity_r_iz.png")

# =========================================================
# 7. БАЗОВАЯ КРИВАЯ ЗОНДИРОВАНИЯ
# =========================================================
znacheniya_L = np.linspace(0.01, 2.0, 500)
krivaya_zondirovaniya = np.array(
    [forward(model, L, sigma=sigma_perehoda) for L in znacheniya_L],
    dtype=float
)

plt.figure(figsize=(10, 6))
plt.plot(
    znacheniya_L,
    krivaya_zondirovaniya,
    "r--",
    linewidth=2,
    label=f"Сглаженная модель (σ={sigma_perehoda})"
)
plt.axhline(y=rho_b, color="gray", linestyle="--", alpha=0.7, label=f"ρ_b = {rho_b} Ом·м")
plt.axhline(y=rho_iz, color="gray", linestyle=":", alpha=0.7, label=f"ρ_из = {rho_iz} Ом·м")
plt.axhline(y=rho_f, color="gray", linestyle="-.", alpha=0.7, label=f"ρ_f = {rho_f} Ом·м")
plt.axvline(x=r_b, color="blue", linestyle="--", alpha=0.5, label=f"r_b = {r_b} м")
plt.axvline(x=r_iz, color="green", linestyle="--", alpha=0.5, label=f"r_из = {r_iz} м")

plt.xlabel("Длина зонда L (м)")
plt.ylabel("Кажущееся удельное сопротивление (Ом·м)")
plt.title("Базовая кривая зондирования")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
sohranit_i_pokazat_grafik("base_sounding_curve.png")

print("Все расчеты завершены успешно.")
print(f"Графики сохранены в папке: {PAPKA_GRAFIKOV}")