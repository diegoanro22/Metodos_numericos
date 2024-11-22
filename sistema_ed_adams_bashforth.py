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

# Implementación de diferencias finitas
# Crear matrices para el sistema lineal
M = np.zeros((2 * n_steps, 2 * n_steps))
b = np.zeros(2 * n_steps)

# Llenar la matriz y el vector para el sistema de ecuaciones
for i in range(n_steps):
    if i == 0:  # Condiciones iniciales
        M[i, i] = 1
        b[i] = y1[0]
        M[i + n_steps, i + n_steps] = 1
        b[i + n_steps] = y2[0]
    elif i == n_steps - 1:  # Último punto (dummy para cierre)
        M[i, i] = 1
        M[i + n_steps, i + n_steps] = 1
    else:  # Puntos internos
        M[i, i - 1] = -B1 / h
        M[i, i] = 2 / h + C1
        M[i, i + 1] = -B1 / h
        b[i] = -A1 * x[i]

        M[i + n_steps, i + n_steps - 1] = -B2 / h
        M[i + n_steps, i + n_steps] = 2 / h + C2
        M[i + n_steps, i + n_steps + 1] = -B2 / h
        b[i + n_steps] = -A2 * x[i]

# Resolver el sistema
y = np.linalg.solve(M, b)
y1_fd = y[:n_steps]
y2_fd = y[n_steps:]

# Mostrar los resultados en formato de texto
print("Resultados (Diferencias Finitas):")
print(f"{'x':>6} {'y1_fd':>10} {'y2_fd':>10}")
for i in range(n_steps):
    print(f"{x[i]:6.2f} {y1_fd[i]:10.5f} {y2_fd[i]:10.5f}")

# Graficar las soluciones
plt.figure(figsize=(12, 8))

# Adams-Bashforth
plt.plot(x, y1, label='y1(x) - Adams-Bashforth', marker='o')
plt.plot(x, y2, label='y2(x) - Adams-Bashforth', marker='s')

# Diferencias Finitas
plt.plot(x, y1_fd, label='y1(x) - Diferencias Finitas', linestyle='--')
plt.plot(x, y2_fd, label='y2(x) - Diferencias Finitas', linestyle='--')

plt.title('Solución del sistema de ED')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
