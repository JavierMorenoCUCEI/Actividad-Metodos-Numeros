import numpy as np
from scipy.optimize import root

# Definición de los 4 sistemas de ecuaciones
def sys1(v):
    x, y = v
    return [x**2 + y**2 - 10, 2*x + y - 1]

def sys2(v):
    x, y = v
    return [x*y - 2, x**2 - 2*y**2 - 2]

def sys3(v):
    x, y = v
    return [x**2 - y**2 + 12, 2*x**2 + y**2 - 24]

def sys4(v):
    x, y = v
    return [x**2 + 3*x + 2 - y, 2*x + 3 - y]

sistemas = [
    ("Sistema i", sys1, [1.0, 1.0], 'hybr'),       # Método híbrido de Powell
    ("Sistema ii", sys2, [2.0, 1.0], 'lm'),        # Levenberg-Marquardt
    ("Sistema iii", sys3, [2.0, 4.0], 'broyden1'), # Método de Broyden (Quasi-Newton)
    ("Sistema iv", sys4, [1.0, 5.0], 'krylov')     # Aproximación de Krylov
]

print("--- EJERCICIO 1: SOLUCIÓN Y ANÁLISIS DE ERROR ---")
for nombre, func, x0, metodo in sistemas:
    sol = root(func, x0, method=metodo)
    residuo = np.array(func(sol.x))
    error_abs = np.linalg.norm(residuo) # Norma euclidiana del residuo (error respecto a 0)
    
    print(f"\n{nombre} (Método: {metodo}):")
    print(f"  Punto (x, y) = ({sol.x[0]:.6f}, {sol.x[1]:.6f})")
    print(f"  Evaluación F(x,y) = [{residuo[0]:.2e}, {residuo[1]:.2e}]")
    print(f"  Error residual (Norma L2) = {error_abs:.2e}")