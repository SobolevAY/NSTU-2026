import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# =========================================================
# 1. МОДЕЛЬ СРЕДЫ
# =========================================================
rho_b = 1.0      # скважина
rho_iz = 30.0    # зона проникновения
rho_f = 10.0    # пласт
r_b = 0.1        # радиус скважины
r_iz = 0.5       # радиус зоны проникновения

model = (rho_b, rho_iz, rho_f, r_b, r_iz)


# =========================================================
# 2. ПРЯМАЯ ЗАДАЧА (СТУПЕНЧАТАЯ)
# =========================================================
def forward_step(model, L):
    """Исходная модель с резкими границами (по площадям сечений)."""
    rho_b, rho_iz, rho_f, r_b, r_iz = model

    if L <= r_b:
        return rho_b
    elif L <= r_iz:
        sigma_eff = (r_b**2 / rho_b + (L**2 - r_b**2) / rho_iz) / L**2
        return 1.0 / sigma_eff
    else:
        sigma_eff = (r_b**2 / rho_b + (r_iz**2 - r_b**2) / rho_iz +
                     (L**2 - r_iz**2) / rho_f) / L**2
        return 1.0 / sigma_eff


# =========================================================
# 3. ПРЯМАЯ ЗАДАЧА (ГЛАДКАЯ) С ИСПОЛЬЗОВАНИЕМ ERF
# =========================================================
def forward_smooth(model, L, sigma=0.5):
    """
    Гладкая модель: веса слоёв определяются функцией ошибок.
    sigma – параметр, управляющий шириной переходной зоны.
    """
    rho_b, rho_iz, rho_f, r_b, r_iz = model

    # Логарифмическая шкала для более естественного перехода по радиусу
    logL = np.log(L)
    log_rb = np.log(r_b)
    log_riz = np.log(r_iz)

    # Веса для трёх слоёв
    w_b = 0.5 * (1 - erf((logL - log_rb) / (sigma * np.sqrt(2))))
    w_iz = 0.5 * (erf((logL - log_rb) / (sigma * np.sqrt(2))) -
                  erf((logL - log_riz) / (sigma * np.sqrt(2))))
    w_f = 0.5 * (1 + erf((logL - log_riz) / (sigma * np.sqrt(2))))

    # Эффективная проводимость как средневзвешенное проводимостей
    sigma_eff = w_b / rho_b + w_iz / rho_iz + w_f / rho_f
    # Защита от деления на ноль при малых L (когда w_b=1, остальные 0)
    return 1.0 / sigma_eff if sigma_eff > 0 else np.inf


# =========================================================
# 4. РАСЧЁТ КРИВЫХ ЗОНДИРОВАНИЯ
# =========================================================
L_values = np.linspace(0.01, 2.0, 500)  # мельче для гладкости

rho_step = [forward_step(model, L) for L in L_values]
rho_smooth = [forward_smooth(model, L) for L in L_values]


# =========================================================
# 5. ВИЗУАЛИЗАЦИЯ (СРАВНЕНИЕ)
# =========================================================
plt.figure(figsize=(10, 6))

# Ступенчатая модель
plt.plot(L_values, rho_step, 'b-', linewidth=2, label='Ступенчатая модель (площадное взвешивание)')

# Гладкая модель
plt.plot(L_values, rho_smooth, 'r--', linewidth=2, label=f'Гладкая модель (erf, σ={0.2})')

# Истинные сопротивления слоёв
plt.axhline(y=rho_b, color='gray', linestyle='--', alpha=0.7, label=f'ρ_скв = {rho_b} Ом·м')
plt.axhline(y=rho_iz, color='gray', linestyle=':', alpha=0.7, label=f'ρ_из = {rho_iz} Ом·м')
plt.axhline(y=rho_f, color='gray', linestyle='-.', alpha=0.7, label=f'ρ_пл = {rho_f} Ом·м')

# Границы радиусов
plt.axvline(x=r_b, color='red', linestyle='--', alpha=0.5, label=f'Радиус скважины = {r_b} м')
plt.axvline(x=r_iz, color='red', linestyle=':', alpha=0.5, label=f'Радиус зоны проникновения = {r_iz} м')

plt.xlabel('Длина зонда L (м)')
plt.ylabel('Кажущееся сопротивление (Ом·м)')
plt.title('Сравнение ступенчатой и гладкой моделей прямой задачи')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()


# =========================================================
# 6. ПРИМЕРЫ ДЛЯ ВЫБРАННЫХ ЗОНДОВ
# =========================================================
probe_lengths = [0.2, 0.5, 1.0]
print("Модель среды:")
print(f"  ρ_скв = {rho_b} Ом·м, r_скв = {r_b} м")
print(f"  ρ_из  = {rho_iz} Ом·м, r_из  = {r_iz} м")
print(f"  ρ_пл  = {rho_f} Ом·м\n")

print("Результаты для выбранных зондов:")
for L in probe_lengths:
    rho_s = forward_step(model, L)
    rho_sm = forward_smooth(model, L)
    print(f"L = {L:.2f} м: ступенчатая = {rho_s:.2f} Ом·м, гладкая = {rho_sm:.2f} Ом·м")