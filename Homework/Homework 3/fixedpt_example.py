# import libraries
import numpy as np
import matplotlib.pyplot as plt
    
# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]
    

# use routines 
x0 = -0.89835
tol = (10**(-11))
Nmax = 1000
f = lambda x: -np.sin(2*x) + 5.0*x/4.0 - 3.0/4.0

[xstar,ier] = fixedpt(f,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f(xstar):',f(xstar))
print('Error message reads:',ier)

f = lambda x: x - 4*np.sin(2*x) - 3

x = np.linspace(-1, 5,10000)
y = f(x)

plt.plot(x,y)
plt.plot(xstar,f(xstar),'ro')
plt.plot(x,[0 for _ in range(len(x))])
plt.show()