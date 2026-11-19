import numpy as np
from scipy.optimize import curve_fit

# Datos del Ejercicio 2
x3 = np.array([
    -4.00000, -3.33333, -2.66667, -2.00000, -1.33333, -0.66667,
     0.00000,  0.66667,  1.33333,  2.00000,  2.66667,  3.33333,  4.00000
])
y3 = np.array([
    3.86799, 3.91298, 4.45325, 4.75713, 4.90094, 5.19899,
    5.43379, 5.52649, 5.05776, 5.18603, 5.09235, 4.38856, 3.75928
])

def mse(y_real, y_pred):
    return np.mean((y_real - y_pred)**2)

# a. Lineal
p_lin = np.polyfit(x3, y3, 1)
mse_lin = mse(y3, np.polyval(p_lin, x3))

# b. Cuadrático y Cúbico
p_cuad = np.polyfit(x3, y3, 2)
mse_cuad = mse(y3, np.polyval(p_cuad, x3))

p_cub = np.polyfit(x3, y3, 3)
mse_cub = mse(y3, np.polyval(p_cub, x3))

# c. Exponencial
def mod_exp(x, a, b):
    return a * np.exp(b * x)
popt_exp, _ = curve_fit(mod_exp, x3, y3, p0=[5, 0.01])
mse_exp = mse(y3, mod_exp(x3, *popt_exp))

# d. Circular: y^2 = a*x^2 + b*x + c
p_circ = np.polyfit(x3, y3**2, 2)
y_pred_circ = np.sqrt(np.maximum(0, np.polyval(p_circ, x3)))
mse_circ = mse(y3, y_pred_circ)

errores3 = {
    "Lineal": mse_lin,
    "Cuadrático": mse_cuad,
    "Cúbico": mse_cub,
    "Exponencial": mse_exp,
    "Circular": mse_circ
}

print("--- EJERCICIO 2: LECTURAS RAMAN ---")
print(f"a) Lineal:      MSE = {mse_lin:.6f}")
print(f"b) Cuadrático:  MSE = {mse_cuad:.6f}")
print(f"   Cúbico:      MSE = {mse_cub:.6f}")
print(f"c) Exponencial: MSE = {mse_exp:.6f}")
print(f"d) Circular:    y^2 = {p_circ[0]:.4f}x^2 + {p_circ[1]:.4f}x + {p_circ[2]:.4f} | MSE = {mse_circ:.6f}")

mejor3 = min(errores3, key=errores3.get)
print(f"\ne) El modelo con menor error es: {mejor3} (MSE = {errores3[mejor3]:.6f})")
