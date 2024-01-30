#%% 3.1
x = [1,2,3]
print(3*x)

import numpy as np

y = np.array([1,2,3])
print(3*y)

import matplotlib.pyplot as plt

X = np.linspace(0, 2*np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

plt.plot(X,Ya)
plt.plot(X,Yb)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#X is size 100 since what linspace does is create an array of 
#linearly spaced (linspace) numbers between the first and second input
#with the third input being the number of steps

#%%3.2
x = np.linspace(0,0.99,100)
y = np.arange(0,1,0.01)

if len(x) == len(y):
    print("x and y have the same length")
else:
    print("x and y do not have the same length")
  
print()
print("The first three entries of x are:")
for i in range(0,3):
    print(x[i],end=", ")

w = 10**(-np.linspace(1,10,10))

#w is a vector that contains the powers of 10 from 10^-1 to 10^-10

s = 3*w
x = np.linspace(1,10,10)

plt.semilogy(x,w)   
plt.semilogy(x,s)
plt.xlabel('x')
plt.ylabel('w')
plt.show()

