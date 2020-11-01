"""
Este script implementa el método de Crank-Nicolson para resolver la ecuacion:

dT/dt = d^2T/dx^2

Con 0 < x < 1.
T(t, x=0) = T(t, x=1) = 0
y 
T(t=0, x) = sin(pi x)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags  # to easily create diagonal matrices
from scipy.linalg import solve  # to solve S @ T = b, solve for T


# Discretizacion del espacio
N = 51  # numero de puntos a evaluar en el espacio
h = 1 / (N-1)  # delta x

x = np.linspace(0, 1, N)
T = np.sin(np.pi * x)  # Condicion inicial, tambien la solucion que buscamos


# Discretizacion del tiempo
epsilon = 0.01  # delta t adimensionalizado, partimos con numero arbitrario
N_temporales = 30

# En el siguiente vector se guardaran las soluciones en cada instante
T_solucion = np.zeros((N_temporales, N))
T_solucion[0] = T


# Definiendo matriz tridiagonal
s = epsilon / 2 / h**2

# shape de la matriz
diagonal = np.ones(N-2) * (2*s + 1)
off_diagonals = np.ones(N-3) * (-s)
S = diags([off_diagonals, diagonal, off_diagonals], 
          offsets=[1, 0, -1]).toarray()

# Para visualizar la matriz S se puede hacer:
# S.toarray()
# pero conviene mantenerla como una "sparse matrix" para eficiencia

# Al lado derecho tambien vamos a necesitar una matriz tri-diagonal para
# calcular los elementos del vector b
diagonal_derecha = np.ones(N-2) * (1 - 2*s)
off_diagonals_derecha = np.ones(N-3) * (s)
S_derecha = diags(
            [off_diagonals_derecha, diagonal_derecha, off_diagonals_derecha],
            offsets=[1, 0, -1])


for i in range(1, N_temporales):
    # Implementando un paso temporal
    # Calculamos el vector b (lado derecho de la ec. de C-N)
    b = S_derecha @ T_solucion[i-1, 1:-1]
    b[0] = b[0] + s * T_solucion[i-1, 0]   # valido para condiciones de borde rigidas
    b[-1] = b[-1] + s * T_solucion[i-1, -1]

    # Ahora el problema a resolver es S @ T = b; despejar T
    T_solucion[i, 1:-1] = solve(S, b)


# Visualizacion de la condicion inicial
plt.figure(1)
plt.clf()

for i in range(0, N_temporales, 3):
    plt.plot(x, T_solucion[i], '-', label='T(t={:.3f}, x)'.format(i * epsilon))

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()
plt.savefig('soluciones-con-mejor-resolucion.png')

