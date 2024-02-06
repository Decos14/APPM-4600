# import libraries
import numpy as np


# define routines
def bisection(f,a,b,tol,Nmax):
    '''
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    '''

    '''     first verify there is a root we can find in the interval '''
    fa = f(a); fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

    ''' verify end point is not a root '''
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    while (count < Nmax):
      c = 0.5*(a+b)
      fc = f(c)

      if (fc ==0):
        astar = c
        ier = 0
        return [astar, ier]

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar, ier]

      if (abs(b-a)<tol):
        astar = a
        ier =0
        return [astar, ier]
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier] 

# use routines    
f = lambda x: (x**2)*(x-1)
a = 0.5
b = 2

Nmax = 100
tol = 1e-3
print("f(x) = (x^2)(x-1)")
print("(a,b) = (0.5,2)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

a = -1
b = 0.5

Nmax = 100
tol = 1e-3
print("f(x) = (x^2)(x-1)")
print("(a,b) = (-1,0.5)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

a = -1
b = 2

Nmax = 100
tol = 1e-3
print("f(x) = (x^2)(x-1)")
print("(a,b) = (-1,2)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

# use routines    
f = lambda x: (x-1)*(x-3)*(x-5)
a = 0
b = 2.4

Nmax = 100
tol = 1e-3
print("f(x) = (x-1)(x-3)(x-5)")
print("(a,b) = (0,2.4)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

# use routines    
f = lambda x: (x-1)*(x-3)*(x-1)
a = 0
b = 2

Nmax = 100
tol = 1e-3
print("f(x) = ((x-1)^2)(x-3)")
print("(a,b) = (0,2)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

# use routines    
f = lambda x: np.sin(x)
a = 0
b = 0.1

Nmax = 100
tol = 1e-3
print("f(x) = sin(x)")
print("(a,b) = (0,0.1)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

f = lambda x: np.sin(x)
a = 0.5
b = 3.0*np.pi/4.0

Nmax = 100
tol = 1e-3
print("f(x) = sin(x)")
print("(a,b) = (0,0.1)")
[astar,ier] = bisection(f,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print()

