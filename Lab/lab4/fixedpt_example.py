# import libraries
import numpy as np
import matplotlib.pyplot as plt

f3 = lambda x: (10.0/(x+4))**(1/2)

x0 = 1
Nmax = 1000
tol = 1e-10

def vector_fixedpt(f,x0,tol,Nmax):

   ''' x0 = initial guess''' 
   ''' Nmax = max number of iterations'''
   ''' tol = stopping tolerance'''
   its = []
   count = 0
   while (count < Nmax):
      count = count +1
      x1 = f(x0)
      if (abs(x1-x0) <tol):
         xstar = x1
         ier = 0
         return [its,ier]
      its.append(x0)
      x0 = x1
   xstar = x1
   its.append(xstar)
   ier = 1
   its.append(1)
   return [its, ier]

def aitkens(seq,tol):
   p_hat = []
   for i in range(len(seq)-2):
      val = seq[i]- (((seq[i+1]-seq[i])**2)/(seq[i+2]-2*seq[i+1]+seq[i]))
      p_hat.append(val)
      if i > 0 and np.abs(p_hat[i]-p_hat[i-1]) < tol:
          break
   return p_hat

pv = vector_fixedpt(f3,x0,tol,Nmax)[0]
print(pv)
p_hat = aitkens(pv,tol)
print(p_hat)

p = 1.3652300134140976

y1 = []
y2 = []
y3 = []
for i in range(len(p_hat)-1):
    y1.append(np.abs(p_hat[i+1]-p)/np.abs(p_hat[i]-p))
    y2.append(np.abs(p_hat[i+1]-p)/np.abs(p_hat[i]-p)**2)
    y3.append(np.abs(p_hat[i+1]-p)/np.abs(p_hat[i]-p)**3)

print(y1)
print(y2)
print(y3)
plt.plot(y1)
plt.show()
plt.plot(y2)
plt.show()
plt.plot(y3)