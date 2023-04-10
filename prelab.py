import numpy as np

def eval_legrange(n,x):
    phi = np.zeros(n+1)
    phi[0] = 1
    phi[1] = x
    for i in range(n-1):
        phi[i+2] = (1/(n+1))*((2*n+1)*x*phi[i+1] - n*phi[i])
    return phi
print(eval_legrange(5,3))