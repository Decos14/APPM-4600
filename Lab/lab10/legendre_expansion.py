import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import scipy as sp

def driver():

#  function you want to approximate
    f1 = lambda x: np.exp(x)
    f2 = lambda x: 1./(1.+x**2)
    

# Interval of interest
    a = -1
    b = 1
# weight function
    w = lambda x: 1.
    wc = lambda x: 1./np.sqrt(1.-x**2)

# order of approximation
    n = 4

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval1 = np.zeros(N+1)
    pval2 = np.zeros(N+1)
    pvalc = np.zeros(N+1)

    for kk in range(N+1):
      pval1[kk] = eval_legendre_expansion(f1,a,b,w,n,xeval[kk])
      pval2[kk] = eval_legendre_expansion(f2,a,b,w,n,xeval[kk])
      pvalc[kk] = eval_chebychev_expansion(f1,a,b,wc,n,xeval[kk])

    ''' create vector with exact values'''
    fex1 = np.zeros(N+1)
    fex2 = np.zeros(N+1)
    fexc = np.zeros(N+1)
    for kk in range(N+1):
        fex1[kk] = f1(xeval[kk])
        fex2[kk] = f2(xeval[kk])
        fexc[kk] = f1(xeval[kk])

    plt.figure();
    plt.plot(xeval,pval1);
    plt.show()

    plt.figure();
    err = abs(pval1-fex1)
    plt.plot(xeval,np.log10(err)); 
    plt.show()

    plt.figure();
    plt.plot(xeval,pval2);
    plt.show()
    
    plt.figure();
    err = abs(pval2-fex2)
    plt.plot(xeval,np.log10(err)); 
    plt.show()
    
    plt.figure();
    plt.plot(xeval,pvalc);
    plt.show()
    
    plt.figure();
    err = abs(pvalc-fexc)
    plt.plot(xeval,np.log10(err)); 
    plt.show()



def eval_legendre_expansion(f,a,b,w,n,x):

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab
  p = [eval_legendre(i, x) for i in range(n+1)]
  # initialize the sum to 0
  pval = 0.0
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_legendre(j,x)
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: phi_j(x)**2*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: phi_j(x)*f(x)*w(x)/norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      # accumulate into pval
      pval = pval+aj*p[j]

  return pval

def eval_chebychev_expansion(f,a,b,w,n,x):

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab
  p = [eval_chebychev(i, x) for i in range(n+1)]
  # initialize the sum to 0
  pval = 0.0
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_chebychev(j,x)
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: phi_j(x)**2*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: phi_j(x)*f(x)*w(x)/norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      # accumulate into pval
      pval = pval+aj*p[j]

  return pval


def eval_legendre(N,x):
    if N == 0:
        return 1
    if N == 1:
        return x
    else:
        return (1./(N))*((2*N-1)*x*eval_legendre(N-1, x) - (N-1)*eval_legendre(N-2, x))
    
def eval_chebychev(N,x):
    if N == 0:
        return 1
    if N == 1:
        return x
    else:
        return 2*x*eval_chebychev(N-1,x) - eval_chebychev(N-2, x)

if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()
