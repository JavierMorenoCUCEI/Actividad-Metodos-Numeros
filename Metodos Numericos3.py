import numpy as np
from scipy.optimize import curve_fit

# Datos del Ejercicio 3
x2 = np.arange(0, 15.5, 0.5)
y2 = np.array([
    4.34578, 4.37718, 6.00581, 6.90632, 8.73317, 10.42939, 13.13911,
    16.42928, 20.45959, 25.12904, 29.49112, 34.76709, 40.99071, 46.41022,
    53.45534, 60.6661, 68.73504, 77.01793, 85.9129, 94.28083, 104.10426,
    114.25524, 125.18293, 136.73854, 148.13832, 160.2527, 173.76404,
    187.08744, 200.7447, 215.15871, 229.34054
])

def mse(y_real, y_pred):
    return np.mean((y_real - y_pred)**2)

# a. Modelo Lineal
p_lin = np.polyfit(x2, y2, 1)
mse_lin = mse(y2, np.polyval(p_lin, x2))

# b. Modelos Cuadrático y Cúbico
p_cuad = np.polyfit(x2, y2, 2)
mse_cuad = mse(y2, np.polyval(p_cuad, x2))

p_cub = np.polyfit(x2, y2, 3)
mse_cub = mse(y2, np.polyval(p_cub, x2))

# c. Modelo Exponencial: y = a * exp(b * x)
def mod_exp(x, a, b):
    return a * np.exp(b * x)

popt_exp, _ = curve_fit(mod_exp, x2, y2, p0=[4, 0.2])
mse_exp = mse(y2, mod_exp(x2, *popt_exp))

# Resultados
errores = {
    "Lineal": mse_lin,
    "Cuadrático": mse_cuad,
    "Cúbico": mse_cub,
    "Exponencial": mse_exp
}

print("--- EJERCICIO 2: ANÁLISIS DE MODELOS ---")
print(f"a) Lineal:      y = {p_lin[0]:.4f}x + {p_lin[1]:.4f} | MSE = {mse_lin:.6f}")
print(f"b) Cuadrático:  y = {p_cuad[0]:.4f}x^2 + {p_cuad[1]:.4f}x + {p_cuad[2]:.4f} | MSE = {mse_cuad:.6f}")
print(f"   Cúbico:      y = {p_cub[0]:.4f}x^3 + {p_cub[1]:.4f}x^2 + {p_cub[2]:.4f}x + {p_cub[3]:.4f} | MSE = {mse_cub:.6f}")
print(f"c) Exponencial: y = {popt_exp[0]:.4f} * e^({popt_exp[1]:.4f}x) | MSE = {mse_exp:.6f}")

mejor_modelo = min(errores, key=errores.get)
print(f"\nd) El modelo con menor Error Cuadrático Medio es: {mejor_modelo} (MSE = {errores[mejor_modelo]:.6f})")