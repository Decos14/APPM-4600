import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm


def driver():
    f = lambda x: 1./(1.+(10.*x)**2)
    a = 0
    b = 1
    '''create points you want to evaluate at'''
    Neval = 100
    xeval = np.linspace(a,b,Neval)
    '''number of intervals'''
    Nint = 5
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    '''evaluate f at the evaluation points'''
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'r')
    plt.plot(xeval,yeval,'b')
    plt.legend()
    plt.show
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show
    
def line(x_0,x_1,a):
    return x_0[1] + (a - x_0[0])*(x_1[1]-x_0[1])/(x_1[0]-x_0[0])

    
def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
        # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
        '''temporarily store your info for creating a line in the interval of
        interest'''
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))
        for kk in range(n):
            #use your line evaluator to evaluate the spline at each location
            yloc[kk] = line((atmp,fa),(btmp,fb),xloc[kk])#Call your line evaluator with points (atmp,fa) and (btmp,fb)
            
        # Copy yloc into the final vector
        yeval[ind] = yloc
    return yeval

driver()

def driver():
    
    f = lambda x: 1./(1.+(10.*x)**2)
    a = 0
    b = 1
    
    
    ''' number of intervals'''
    Nint = 5
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

    
    
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    
    print('M =', M)
#    print('C =', C)
#    print('D=', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'r',label='exact function')
    plt.plot(xeval,yeval,'b',label='natural spline') 
    plt.legend
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0,0] = 1
    A[N,N] = 1
    for i in range(1,N):
        for j in range(1,N):
            if j == i - 1:
                A[i,j] =  h[i-1]
            if j == i:
                A[i,j] =  2*(h[i]+h[i-1])
            if j == i + 1:
                A[i,j] =  h[i]
                

    Ainv = inv(A)
    
    M  = np.matmul(Ainv,b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 + (xeval-xi)**3*Mip)/(6*hi) + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()          