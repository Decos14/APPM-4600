# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:18:55 2024

@author: devli
"""
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
    return [xstar, ier,count]

def bisection(f,a,b,tol,Nmax,conv):
    '''
    Inputs:
      f,a,b       - function and endpoints of initial interval
      conv, Nmax  - bisection stops when the midpoint is in conv
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
      mid = (fa+fb)/fc

      if (fc ==0):
        astar = c
        ier = 0
        return [astar,ier,count]    

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar,ier,count]

      if mid > conv[0] and mid < conv[1]:
        astar = a
        ier =0
        fixed = fixedpt(f,mid,tol,Nmax-count)
        fixed[2] += count
        return fixed
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier,count] 