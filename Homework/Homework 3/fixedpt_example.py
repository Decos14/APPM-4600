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
x0 = 2.5
tol = (10**(-10))
Nmax = 30
f = lambda x: x - (((x**2)+1)/(2*x))

[xstar,ier] = fixedpt(f,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f(xstar):',f(xstar))
print('Error message reads:',ier)


x = np.linspace(1, 4,10000)
y = f(x)

plt.plot(x,y)
plt.plot(xstar,f(xstar),'ro')
plt.show()