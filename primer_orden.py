import numpy as np
import matplotlib.pyplot as plt

# Definición de la función de la ecuación diferencial
def f(x, y):
    return (1 - x - y) / (x + y)

# Método de Adams-Bashforth
def adams_bashforth(x0, y0, x_end, h):
    n = int((x_end - x0) / h)  # Número de puntos
    x = np.linspace(x0, x_end, n + 1)
    y = np.zeros(n + 1)
    
    # Condición inicial
    y[0] = y0
    
    # Método de Euler para obtener el segundo punto
    y[1] = y[0] + h * f(x[0], y[0])
    
    # Adams-Bashforth de dos pasos
    for i in range(1, n):
        y[i + 1] = y[i] + h / 2 * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
    
    return x, y

# Método de diferencias finitas
def diferencias_finitas(x0, y0, x_end, h):
    n = int((x_end - x0) / h)  # Número de puntos
    x = np.linspace(x0, x_end, n + 1)
    y = np.zeros(n + 1)
    
    # Condición inicial
    y[0] = y0
    
    # Diferencias finitas explícitas
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    
    return x, y

# Parámetros iniciales
x0 = 1.0  # Valor inicial de x
y0 = 1.0  # Valor inicial de y
x_end = 2.0  # Valor final de x
h = 0.1  # Tamaño de paso

# Resolver la ecuación con ambos métodos
x_adams, y_adams = adams_bashforth(x0, y0, x_end, h)
x_diff, y_diff = diferencias_finitas(x0, y0, x_end, h)

# Mostrar los puntos calculados
print("Puntos calculados por el método de Adams-Bashforth:")
print("x\t\ty")
for xi, yi in zip(x_adams, y_adams):
    print(f"{xi:.2f}\t{yi:.5f}")

print("\nPuntos calculados por el método de Diferencias Finitas:")
print("x\t\ty")
for xi, yi in zip(x_diff, y_diff):
    print(f"{xi:.2f}\t{yi:.5f}")

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(x_adams, y_adams, 'o-', label="Adams-Bashforth", markersize=8)
plt.plot(x_diff, y_diff, 's-', label="Diferencias Finitas", markersize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la Ecuación Diferencial")
plt.legend()
plt.grid(True)
plt.show()
