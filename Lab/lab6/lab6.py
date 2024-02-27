import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt;
def driver():
    x0 = np.array([1., 0.])
    Nmax = 100
    tol = 1e-10
    t = time.time() 
    for j in range(20):
        [xstar,newtxlist,ier,its] = Newton(x0,tol,Nmax);
    elapsed = time.time()-t;
    print("Newton:" + str(xstar));
    err = np.sum((newtxlist-xstar)**2,axis=1);
    plt.plot(np.arange(its),np.log10(err[0:its]), label = "Newton");
    plt.title("Newton Methods")
    plt.xlabel("iterations")
    plt.ylabel("log10(errors)")
    
    t = time.time()
    for j in range(20):
        [xstar,lazyxlist,ier,its] = LazyNewton(x0,tol,Nmax);
    elapsed = time.time()-t
    print("Lazy Newton" + str(xstar));
    err2 = np.sum((lazyxlist-xstar)**2,axis=1);
    plt.plot(np.arange(its),np.log10(err2[0:its]),label = "Lazy Newton");
    
    t = time.time()
    for j in range(20):
        [xstar,slackxlist,ier,its,updates] = SlackerNewton(x0,tol,Nmax);
    elapsed = time.time()-t
    print("Slacker Newton" + str(xstar));
    err2 = np.sum((slackxlist-xstar)**2,axis=1);
    plt.plot(np.arange(its),np.log10(err2[0:its]), label = "Slacker Newton");
    plt.legend()
    plt.show()
    print(updates)
    new = np.array(newtxlist)
    newx = new[:,0]
    newy = new[:,1]
    plt.scatter(newx,newy)
def evalF(x):
    F = np.zeros(2)
    F[0] = 4.*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0]-x[1])
    return F
def evalJ(x):
    J = np.array([[8*x[0],2*x[1]],
                 [1-np.cos(x[0]-x[1]),1-np.cos(x[0]-x[1])]])
    return J
def Newton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;
    for its in range(Nmax):
        J = evalJ(x0);
        F = evalF(x0);
        x1 = x0 - np.linalg.solve(J,F);
        xlist[its+1]=x1;
        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier =0
            return[xstar, xlist,ier, its];
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its];
def LazyNewton(x0,tol,Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;
    J = evalJ(x0);
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - np.linalg.solve(J,F);
        xlist[its+1]=x1;
        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier =0
            return[xstar,xlist, ier,its];
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its];

def SlackerNewton(x0,tol,Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;
    J = evalJ(x0);
    updates = 0
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - np.linalg.solve(J,F);
        xlist[its+1]=x1;
        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier =0
            return[xstar,xlist, ier,its,updates];
        if np.linalg.norm(evalF(x1)) <= np.linalg.norm(evalF(x0)):
            J = evalJ(x0);
            updates += 1
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its, updates];

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver();