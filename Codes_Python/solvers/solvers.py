import numpy as np
import numpy.linalg as linalg

# Proximal operator of l1 norm in closed form
def l1_proximal(x, gamma):
    # also referred to as shrinkage or soft thresholding operator
    return np.multiply(np.sign(x), np.maximum(np.abs(x)-gamma, 0))

# Alternating Direction Method of Multipliers (ADMM) for l1 regularized linear least squares
def admm_l1(A, b, L, x, rho, iterations, cgls_iterations):
    sqrt2rho = np.sqrt(2.0/rho)
        
    x_cur = x.copy()
    y_cur = np.zeros(L.shape[0])
    u_cur = np.zeros(L.shape[0])
    
    bigMat = np.vstack([sqrt2rho*A, L])
    
    for i in range(iterations):
        bigVec = np.hstack([sqrt2rho*b, y_cur - u_cur])
        x_new = cgls(bigMat, bigMat.T, bigVec, x_cur, cgls_iterations)
        
        y_new = l1_proximal(L@x_new + u_cur, 1.0/rho)
        u_new = u_cur + (L@x_new - y_new)

        x_cur, y_cur, u_cur = x_new, y_new, u_new
        
    return x_cur
        
# Conjugate gradient least squares
def cgls(A, AT, b, x, iterations):
    call = callable(A)
    
    r = b - A(x) if call else b-A@x 
    s = AT(r) if call else AT@r

    p = s.copy()
    gamma = linalg.norm(s)**2
        
    for i in range(iterations):
        q = A(p) if call else A@p
            
        delta = linalg.norm(q)**2
        alpha = gamma/delta
            
        x += alpha*p
        r -= alpha*q
        s = AT(r) if call else A.T@r
            
        gamma_old = gamma
        gamma = linalg.norm(s)**2
        p = s + (gamma/gamma_old)*p
        
    return x