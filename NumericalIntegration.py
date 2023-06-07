from math import cos, exp, pi
from scipy.integrate import quad
import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt





# The kernel's definition

def q(x,y,s,t):
    return  4*x*y  - (8*x*y - 4*x*x*exp(s)- 4*y*y*exp(s))/(1-exp(2*s)) + (exp(-s) - exp(s))

def p(x,y,s,t):
    return (1-1/t) * exp(-s) - exp(s) - 4*t*x*y + 4*x*y

def k(x,y,s,t):
    return p(x,y,s,t) / q(x,y,s,t)

# Plot the kernel function

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
# plt.show()








# The weight function:
def w(y,t):
    return 1 / (2*pi) * (4*t*sqrt(1-y*y)) / (1- 4*(1-t)*t*y*y)

def w2(y):
    return w(y,3/4)







# I want to define my polynomials

def Q(x,n,t):
    if n == 0:
        return 1
    elif n == 1:
        return 2*sqrt(t)*x
    elif n==2:
        return 4*sqrt(t)*x**2 - 1 / sqrt(t)
    else:
        return 2*x*Q(x,n-1,t) - Q(x,n-2,t)
    
def Q2(x,n):
    return Q(x,n,3/4)

"""
# TEST: are my polynomials orthogonal?

x = np.linspace(-1, 1, 80)

def integral_polynomials(n):
    return quad(lambda x: Q2(x,n)*w2(x), -1, 1)
for n in range(5):
    print("Integral of polynomial %s is %s" % (n ,integral_polynomials(n)))

def orthogonal_polynomials(n,m):
    return quad(lambda x: Q2(x,n)*Q2(x,m)*w2(x), -1, 1)
for n in range(5):
    print("Polynomials %s the next one gives %s" % (n ,orthogonal_polynomials(n,n+1)))
"""


# I want to see if k acts as I want

x_vals = np.linspace(-1, 1, 80)

def TQ2(x,n,s): # This gives values of the transform of Qn at time s
    def integrand(y,x,n,s):
        return Q2(y,n)*w2(y)*k(x,y,s,3/4)
    return  quad(integrand, -1, 1, args=(x,n,s))[0]


def integral_operator(f, x):
    # Define the integrand as a function of t
    integrand = lambda t: kernel(x, t) * f(t)
    # Evaluate the integral using quad
    return quad(integrand, -np.inf, np.inf)[0]


#print(np.array([TQ2(x,0,1) for x in x_vals]))

def bad_kernel(n,s):
    ArTQ2 = np.array([TQ2(x,n,s) for x in x_vals])
    return max(abs(ArTQ2 - exp(-n*s)*Q2(x,n)))

print(bad_kernel(2,7))

