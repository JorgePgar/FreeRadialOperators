import numpy as np
from matplotlib import pyplot as plt
from math import exp, pi



# Now I try to write my functions. I prepare here to plot my kernel.

def q(x,y,s,t):
    return  4*x*y  - (8*x*y 
                      - 4*x*x*exp(s) 
                      - 4*y*y*exp(s))/(1-exp(2*s))  + (exp(-s) - exp(s))

def p(x,y,s,t):
    return -1/t * exp(-s) - 4*t*x*y + 4*x*y

def k(x,y,s,t):
    return p(x,y,s,t) / q(x,y,s,t)

# Ploting t = 3/4

x = np.linspace(-1, 1, 80)
y = np.linspace(-1, 1, 80)

X, Y = np.meshgrid(x, y)
Z = k(X,Y,1,3/4)

fig = plt.figure(1)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, 
                rstride=1, cstride=1, 
                cmap='viridis', edgecolor='none')
ax.set_title('k_t with t=3/4')

# Ploting t = 1/2

Z2 = k(X,Y,1,1/2)

fig = plt.figure(2)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z2, 
                rstride=1, cstride=1, 
                cmap='viridis', edgecolor='none')
ax.set_title('k_t with t=1/2') 


# Ploting usual kernel

Z2 = k(X,Y,1,1/2)

fig = plt.figure(2)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z2, 
                rstride=1, cstride=1, 
                cmap='viridis', edgecolor='none')
ax.set_title('k_t with t=1/2') 


plt.show()


# Comparing the 

