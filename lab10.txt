import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy import integrate



def eval_legrange(n,x):
    phi = np.zeros(n+1)
    phi[0] = 1
    phi[1] = x
    for i in range(n-1):
        phi[i+2] = (1/(n+1))*((2*n+1)*x*phi[i+1] - n*phi[i])
    return phi

def eval_legendre_expansion(f,a,b,w,n,x):
    # This subroutine evaluates the Legendre expansion
    # Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_legrange(n,x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0,n-1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: (1/(n+1))*((2*n+1)*x*p[j+1] - n*p[j])
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x)**2
        # use the quad function from scipy to evaluate normalizations
        norm_fac = lambda x: phi_j_sq(x) * w(x)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x)*f(x)*w(x)

        ok,wow = integrate.quad(func_j,a,b)

        hmm,sure = integrate.quad(norm_fac,a,b)
        # use the quad function from scipy to evaluate coeffs
        aj = ok / hmm
        # accumulate into pval
        pval = pval+aj*p[j]
    return pval

# function you want to approximate
f = lambda x: math.exp(x)
# Interval of interest
a = -1
b = 1
# weight function
w = lambda x: 1.
# order of approximation
n = 2
# Number of points you want to sample in [a,b]
N = 1000
xeval = np.linspace(a,b,N+1)
pval = np.zeros(N+1)
for kk in range(N+1):
    pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
''' create vector with exact values'''
fex = np.zeros(N+1)
for kk in range(N+1):
    fex[kk] = f(xeval[kk])
plt.figure();
plt.plot(xeval,pval);
plt.show()
plt.figure();
err = abs(pval-fex)
plt.plot(xeval,np.log10(err));
plt.show()