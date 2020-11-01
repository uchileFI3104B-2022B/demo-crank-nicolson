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


# Discretizacion del espacio
N = 5  # numero de puntos a evaluar en el espacio
h = 1 / (N-1)  # delta x

x = np.linspace(0, 1, N)


# Definiendo matriz tridiagonal
epsilon = 0.1  # delta t adimensionalizado, partimos con numero arbitrario
s = epsilon / 2 / h**2

# shape de la matriz
diagonal = np.ones(N-2) * (2*s + 1)
off_diagonals = np.ones(N-3) * (-s)
S = diags([off_diagonals, diagonal, off_diagonals], offsets=[1, 0, -1])

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



# Visualizacion de la condicion inicial
plt.figure(1)
plt.clf()
plt.plot(x, np.sin(np.pi * x), '-', label='T(t=0, x)')

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()

