import numpy as np
import matplotlib.pyplot as plt

# Definimos las constantes del sistema
A1, B1, C1 = 1, -2, 1
A2, B2, C2 = -1, 1, -2

# Definimos las funciones del sistema
def f1(x, y1, y2):
    return -(A1 * x + B1 * y1 + C1 * y2)

def f2(x, y1, y2):
    return -(A2 * x + B2 * y1 + C2 * y2)

# Parámetros del método
h = 0.1  # tamaño de paso
x0, x_end = 0, 2  # intervalo de integración
n_steps = int((x_end - x0) / h) + 1

# Condiciones iniciales
x = np.linspace(x0, x_end, n_steps)
y1 = np.zeros(n_steps)
y2 = np.zeros(n_steps)
y1[0], y2[0] = 1, 0  # y1(0) = 1, y2(0) = 0

# Paso 1: Usar Euler para calcular el primer paso
y1[1] = y1[0] + h * f1(x[0], y1[0], y2[0])
y2[1] = y2[0] + h * f2(x[0], y1[0], y2[0])

# Paso 2: Método de Adams-Bashforth de dos pasos
for i in range(1, n_steps - 1):
    y1[i + 1] = y1[i] + h / 2 * (3 * f1(x[i], y1[i], y2[i]) - f1(x[i - 1], y1[i - 1], y2[i - 1]))
    y2[i + 1] = y2[i] + h / 2 * (3 * f2(x[i], y1[i], y2[i]) - f2(x[i - 1], y1[i - 1], y2[i - 1]))

# Mostrar los resultados en formato de texto
print("Resultados:")
print(f"{'x':>6} {'y1':>10} {'y2':>10}")
for i in range(n_steps):
    print(f"{x[i]:6.2f} {y1[i]:10.5f} {y2[i]:10.5f}")

# Graficar las soluciones
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='y1(x)', marker='o')
plt.plot(x, y2, label='y2(x)', marker='s')
plt.title('Solución del sistema de ED usando Adams-Bashforth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
