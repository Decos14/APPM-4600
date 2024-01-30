import numpy as np
import numpy.linalg as la
import math
import time
def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 0.0*x
    y = f(x)
    w = g(x)
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product is : ', dp)
    return

def dotProduct(x,y,n):
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp
driver()

A = np.array([[1,2],
              [3,4]])
x = np.array([1,2])

def matvec(A,x):
    b = []
    for i in range(len(A)):
        b.append(dotProduct(A[i], x, len(x)))
    return (np.array(b))

print(matvec(A,x))
print(np.matmul(A,x))
print()

C = []
y = []
n = 100
for i in range(n):
    y.append(i)
    temp = []
    for j in range(n):
        temp.append(j)
    C.append(temp)
    
tp0 = time.time()
print(matvec(C,y))
tp1 = time.time()
tn0 = time.time()
print(np.matmul(C,y))
tn1 = time.time()
print()
print("My code took " , tp1-tp0 , "seconds to run")
print("numpy took " , tn1-tn0 , "seconds to run")
print("So numpy is roughly", (tp1-tp0)/(tn1-tn0), "times faster for",n,"x",n,"matrices")