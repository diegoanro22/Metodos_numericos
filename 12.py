import numpy as np
import matplotlib.pyplot as plt

# Parámetros del dominio
a, b = 1, 10  # Intervalo [1, 10]
N = 100  # Número de puntos
x = np.linspace(a, b, N)
h = (b - a) / (N - 1)  # Paso

# Condiciones iniciales
y_a = 2  # y(1) = 2
dy_a = 0  # y'(1) (valor inicial de la derivada)

# Constantes para la solución analítica
C1 = 2
C2 = (5 - 2 * np.cos(np.log(10))) / np.sin(np.log(10))

# Solución analítica
def analytical_solution(x):
    return C1 * np.cos(np.log(x)) + C2 * np.sin(np.log(x))


# Método de Diferencias Finitas
def finite_differences(x, y_a, y_b, N):
    h = x[1] - x[0]
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Condiciones de frontera
    A[0, 0] = 1
    b[0] = y_a
    A[-1, -1] = 1
    b[-1] = y_b

    # Algoritmo de diferencias finitas obtenido de GeeksForGeeks
    for i in range(1, N - 1):
        xi = x[i]
        A[i, i - 1] = xi**2 / h**2 - xi / (2 * h)
        A[i, i] = -2 * xi**2 / h**2 + 1
        A[i, i + 1] = xi**2 / h**2 + xi / (2 * h)
        b[i] = 0

    # Resolución de sistema lineal con una función de python
    y = np.linalg.solve(A, b)
    return y

# Método de Adams-Bashforth
def adams_bashforth(x, y_a, dy_a, N):
    y = np.zeros(N)
    dy = np.zeros(N)

    # Condiciones iniciales del PVI 
    y[0] = y_a
    dy[0] = dy_a

    # Ecuación diferencial: x^2 y'' + x y' + y = 0
    def f(x, y, dy):
        return (-y - x * dy) / x**2

    # Usar el método de Adams-Bashforth (2 pasos)
    for i in range(1, N):
        if i == 1:  # Usar método de Euler para volver la ecuación de segundo orden en primero orden esto basandose en la forma de 2 pasos de Jain(2018)
            dy[i] = dy[i - 1] + h * f(x[i - 1], y[i - 1], dy[i - 1])
            y[i] = y[i - 1] + h * dy[i - 1]
        else:  # Usar Adams-Bashforth utilizando el algoritmo mostrado en ajer.org
            dy[i] = dy[i - 1] + h * (3 * f(x[i - 1], y[i - 1], dy[i - 1]) - f(x[i - 2], y[i - 2], dy[i - 2])) / 2
            y[i] = y[i - 1] + h * (3 * dy[i - 1] - dy[i - 2]) / 2

    return y, dy

# llamadas a las funciones para obtener los datos para las graficas
y_finite = finite_differences(x, y_a, analytical_solution(b), N)
y_adams, dy_adams = adams_bashforth(x, y_a, dy_a, N)
y_analytical = analytical_solution(x)

# Graficar las soluciones de y(x) en gráficos separados
plt.figure(figsize=(18, 6))

# Gráfica 1: Diferencias Finitas
plt.subplot(1, 3, 1)
plt.plot(x, y_finite, 'o-', label='Diferencias Finitas', color='blue', markersize=4)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Método de Diferencias Finitas')
plt.legend()
plt.grid()

# Gráfica 2: Adams-Bashforth
plt.subplot(1, 3, 2)
plt.plot(x, y_adams, 'o-', label='Adams-Bashforth', color='green', markersize=4)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Método Adams-Bashforth')
plt.legend()
plt.grid()

# Gráfica 3: Solución Analítica
plt.subplot(1, 3, 3)
plt.plot(x, y_analytical, '-', label='Solución Analítica', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Solución Analítica')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

