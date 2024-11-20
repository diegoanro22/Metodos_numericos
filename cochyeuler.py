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

    # Condiciones iniciales: y(1) = 2 y y'(1) = 5
    A[0, 0] = 1
    b[0] = 2  # y(1) = 2

    A[1, 0] = -1 / (2 * h)
    A[1, 1] = 1 / (2 * h)
    b[1] = 5  # y'(1) = 5

    # Llenar la matriz para las ecuaciones interiores (i = 2 a N-2)
    for i in range(2, N-1):
        xi = x[i]
        # Sistema basado en la ecuación x^2 y'' + x y' + y = 0
        A[i, i-1] = xi**2 / h**2 - xi / (2 * h)
        A[i, i] = -2 * xi**2 / h**2 - 1
        A[i, i+1] = xi**2 / h**2 + xi / (2 * h)
        b[i] = 0

    A[-1, -1] = 1  # Para evitar singularidades
    b[-1] = 0

    # Resolver el sistema
    y = np.linalg.solve(A, b)
    return y

# Método de Adams-Bashforth para resolver y' = f(x, y)
def ode_system(x, y):
    return [y[1], (-y[0] - x * y[1]) / x**2]

def adams_bashforth(x):
    # Método numérico para resolver el sistema
    sol = solve_ivp(ode_system, [x[0], x[-1]], [2, 5], t_eval=x, method='RK45')
    return sol.y[0]

# Resolver con cada método
y_finite = finite_differences(x, N)
y_adams = adams_bashforth(x)
y_analytical = analytical_solution(x)

# Graficar los resultados de puntos (para cada uno de los métodos)
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

# Graficar la diferencia entre los resultados
# Graficamos las diferencias de cada método con la solución analítica
diff_finite = np.abs(y_finite - y_analytical)
diff_adams = np.abs(y_adams - y_analytical)

# Gráfica de barras para mostrar las diferencias
plt.figure(figsize=(10, 6))
plt.bar(x, diff_finite, width=0.02, label='Diferencias finitas', alpha=0.6)
plt.bar(x, diff_adams, width=0.02, label='Adams-Bashforth', alpha=0.6)
plt.xlabel('x')
plt.ylabel('Diferencia')
plt.title('Diferencia entre resultados numéricos y solución analítica')
plt.legend()
plt.grid()
plt.show()
