import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt

from problem import gen_fin_diff, create_problem
from solvers import admm_l1

"""
Code for solving ||Ax-b||_2^2 + ||Lx||_1
using the Alternating Direction Method of Multipliers (ADMM) method
"""

# Create test problem
n = 128
(x_space, A, b_noisy, x_true) = create_problem(0.02, 10000, 42, n=n)

# Set-up regularization matrix
L = gen_fin_diff(n)

# Compute Tikhonov regularized solution for comparison
Lscal = 2*L
ref_sol = linalg.lstsq(
    np.vstack([A, Lscal]),
    np.hstack([b_noisy,np.zeros((n-1))])
    )[0]

# Compute l1 regularized solution
Lscal = 0.2*L
sol = admm_l1(A, b_noisy, Lscal,
           ref_sol,
           100, # tuning parameter of ADMM
           100, # number of iterations of ADMM
           100) # number of iterations of the inner CGLS algorithm

# Plot data and solutions
fig, axs = plt.subplots(3, figsize = (6, 9))

axs[0].scatter(x_space, x_true, s = 2, c='k')
axs[0].scatter(x_space, b_noisy, s = 2, c='r')
axs[0].set(xlabel='x')
axs[0].legend(['true', 'data'])

axs[1].scatter(x_space, x_true, s = 2, c='k')
axs[1].scatter(x_space, ref_sol, s = 2, c='b')
axs[1].set(xlabel='x')
axs[1].legend(['true', 'Tikhonov'])

axs[2].scatter(x_space, x_true, s = 2, c='k')
axs[2].scatter(x_space, sol, s = 2, c='r')
axs[2].set(xlabel='x')
axs[2].legend(['true', 'TV'])

plt.tight_layout()
