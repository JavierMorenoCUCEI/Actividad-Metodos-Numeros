import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Función original
def f(x):
    return np.exp(x**2 - 2*x + 1)

# Grado 5 requiere 6 puntos (nodos) en el intervalo [-2.5, 2]
x_nodos = np.linspace(-2.5, 2.0, 6)
y_nodos = f(x_nodos)

# a) Polinomio de Interpolación de Lagrange
poly_lagrange = lagrange(x_nodos, y_nodos)

# b) Polinomio de Interpolación de Newton (Diferencias Divididas)
def newton_coef(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    return coef

def eval_newton(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

coef_newton = newton_coef(x_nodos, y_nodos)

# Evaluación en una malla fina para graficar
x_eval = np.linspace(-2.5, 2.0, 200)
y_real = f(x_eval)
y_lag = poly_lagrange(x_eval)
y_new = eval_newton(coef_newton, x_nodos, x_eval)

print("--- EJERCICIO 4: INTERPOLACIÓN POLINOMIAL ---")
print("Polinomio de Lagrange (potencias descendentes):")
print(poly_lagrange)
print("\nCoeficientes de diferencias divididas de Newton:")
print(coef_newton)

# c) Diferencia entre ambas aproximaciones
diff_max = np.max(np.abs(y_lag - y_new))
print(f"\nc) Diferencia máxima absoluta entre Lagrange y Newton: {diff_max:.2e}")
# Gráficas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x_eval, y_real, 'k-', linewidth=2, label='$f(x) = e^{x^2-2x+1}$')
ax1.plot(x_eval, y_lag, 'r--', linewidth=2, label='Lagrange (Grado 5)')
ax1.scatter(x_nodos, y_nodos, color='blue', zorder=5, label='Nodos')
ax1.set_title('a) Interpolación de Lagrange')
ax1.legend()
ax1.grid(True)

ax2.plot(x_eval, y_real, 'k-', linewidth=2, label='$f(x) = e^{x^2-2x+1}$')
ax2.plot(x_eval, y_new, 'g--', linewidth=2, label='Newton (Grado 5)')
ax2.scatter(x_nodos, y_nodos, color='blue', zorder=5, label='Nodos')
ax2.set_title('b) Interpolación de Newton')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
