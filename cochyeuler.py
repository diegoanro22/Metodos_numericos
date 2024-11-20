import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros generales
a, b = 1, 10  # Intervalo del dominio
N = 1000  # Número de puntos
x = np.linspace(a, b, N)  # Puntos del dominio
h = (b - a) / (N - 1)  # Paso

# Solución analítica
def analytical_solution(x):
    C1 = 2
    C2 = 5
    return C1 * np.cos(np.log(x)) + C2 * np.sin(np.log(x))

# Método de diferencias finitas
def finite_differences(x, N):
    h = x[1] - x[0]
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Condiciones iniciales: y(1) = 2 y y'(1) = 5 del PVI, aplicados a los puntos en la frontera
    A[0, 0] = 1
    b[0] = 2  # y(1) = 2

    A[1, 0] = -1 / (2 * h)
    A[1, 1] = 1 / (2 * h)
    b[1] = 5  # y'(1) = 5

    # Llenar la matriz para las ecuaciones interiores (i = 2 a N-2)
    for i in range(2, N-1):
        xi = x[i]
        #x^2 y'' + x y' + y = 0
        A[i, i-1] = xi**2 / h**2 - xi / (2 * h)
        A[i, i] = -2 * xi**2 / h**2 - 1
        A[i, i+1] = xi**2 / h**2 + xi / (2 * h)
        b[i] = 0

    A[-1, -1] = 1  #Esto evita errores en la matriz, obtenido de stackoverflow la solución del error de matrices
    b[-1] = 0

    # Resuelve el sistema con una función
    y = np.linalg.solve(A, b)
    return y

# Método de Adams-Bashforth
def adams_bashforth(x, h):
    y = np.zeros(N)
    dy = np.zeros(N)

    # Condiciones iniciales del PVI
    y[0] = 2
    dy[0] = 5

    # Ecuación diferencial: x^2 y'' + x y' + y = 0
    def f(x, y, dy):
        return (-y - x * dy) / x**2

    # Usar método de Euler para volver la ecuación de segundo orden en primero orden esto basandose en la forma de 2 pasos de Jain(2018)
    dy[1] = dy[0] + h * f(x[0], y[0], dy[0])
    y[1] = y[0] + h * dy[0]

     # Usar Adams-Bashforth utilizando el algoritmo mostrado en ajer.org
    for i in range(1, N - 1):
        dy[i + 1] = dy[i] + h * (3 * f(x[i], y[i], dy[i]) - f(x[i - 1], y[i - 1], dy[i - 1])) / 2
        y[i + 1] = y[i] + h * (3 * dy[i] - dy[i - 1]) / 2

    return y

# Llamada de funciones para cada metodo
y_finite = finite_differences(x, N)
y_adams = adams_bashforth(x,h)
y_analytical = analytical_solution(x)

# Graficar los resultados de puntos
plt.figure(figsize=(15, 5))

# Gráfica 1: Método de diferencias finitas
plt.subplot(1, 3, 1)
plt.plot(x, y_finite, 'o', label='Diferencias finitas', color='blue')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Método de Diferencias Finita')
plt.legend()
plt.grid()

# Gráfica 2: Método de Adams-Bashforth
plt.subplot(1, 3, 2)
plt.plot(x, y_adams, 'o', label='Adams-Bashforth', color='green')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Método Adams-Bashforth')
plt.legend()
plt.grid()

# Gráfica de la solución analítica
plt.subplot(1, 3, 3)
plt.plot(x, y_analytical, 'o', label='Solución analítica', color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Solución Analítica')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

