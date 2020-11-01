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


# Discretizacion del espacio
N = 5  # numero de puntos a evaluar en el espacio
h = 1 / (N-1)  # delta x

x = np.linspace(0, 1, N)


# Visualizacion de la condicion inicial
plt.figure(1)
plt.clf()
plt.plot(x, np.sin(np.pi * x), '-', label='T(t=0, x)')

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()

