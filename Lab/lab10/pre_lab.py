def eval_legendre(N,x):
    if N == 0:
        return 1
    if N == 1:
        return x
    else:
        return (1./(N))*((2*N-1)*x*eval_legendre(N-1, x) - (N-1)*eval_legendre(N-2, x))