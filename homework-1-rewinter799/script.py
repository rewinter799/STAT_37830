"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""
#%%
from matlib import *
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt
from scipy.stats import norm

import time # to compute runtimes
from tqdm import tqdm # progress bar

# # Problem 0

# #%%
# ### Problem 0(A): Construct one hundred 1000x1000 symmetric matrices whose
# ###               entries are standard normal RVs. Compute the eigenvalues
# ###               and plot a histogram. Repeat for n = 200, 400, 800, 1600.
# ###               Guess the distribution.

#%%
# 200 x 200 Symmetric Matrices
evals_200 = eval_simulation(200)
graph_hist_evals(evals_200)

# Label
text_kwargs = dict(ha='center', va='top')
height = np.max(np.histogram(evals_200, density = True, bins = 20)[0]) / 2
plt.text(0, height, "n = 200", text_kwargs)

plt.show()

#%%
# 400 x 400 Symmetric Matrices
evals_400 = eval_simulation(400)
graph_hist_evals(evals_400)

#Label
text_kwargs = dict(ha='center', va='top')
height = np.max(np.histogram(evals_400, density = True, bins = 20)[0]) / 2
plt.text(0, height, "n = 400", text_kwargs)

plt.show()

#%%
# 800 x 800 Symmetric Matrices
evals_800 = eval_simulation(800)
graph_hist_evals(evals_800)

# Label
text_kwargs = dict(ha='center', va='top')
height = np.max(np.histogram(evals_800, density = True, bins = 20)[0]) / 2
plt.text(0, height, "n = 800", text_kwargs)

plt.show()

#%%
# 1000 x 1000 Symmetric Matrices
evals_1000 = eval_simulation(1000)
graph_hist_evals(evals_1000)

# Label
text_kwargs = dict(ha='center', va='top')
height = np.max(np.histogram(evals_1000, density = True, bins = 20)[0]) / 2
plt.text(0, height, "n = 1000", text_kwargs)

plt.show()

#%%
# 1600 x 1600 Symmetric Matrices
evals_1600 = eval_simulation(1600)
graph_hist_evals(evals_1600)

# Label
text_kwargs = dict(ha='center', va='top')
height = np.max(np.histogram(evals_1600, density = True, bins = 20)[0]) / 2
plt.text(0, height, "n = 1600", text_kwargs)

plt.show()


#%%
### Problem 0(B): Generate 1,000 200x200 symmetric matrices whose entries 
###               are independent N(0,1) random variables. Compute the
###               largest eigenvalue of each and plot the histogram. Guess
###               the distribution.

# Initialize empty array of eigenvalues
evals = np.array([])

# Generate 1000 200x200 matrices with N(0,1) entries and save the largest
# eigenvalue of each.
for i in range(1000):
    A = np.random.randn(200, 200)
    # Make randomly-generated matrix A symmetric
    for j in range(200):
        for k in range(j+1, 200):
            A[j, k] = A[k, j]
    # Calculate eigenvalues and add max to running list
    lambdas = sla.eigh(A, eigvals_only = True)
    max_eval = np.max(lambdas)
    evals = np.append(evals, max_eval)

## Histogram
# Labels
plt.xlabel("Maximum Eigenvalues")
plt.ylabel("Frequency")
plt.title("Histogram of Maximum Eigenvalues of 200 x 200 Matrices\
           \nwith Entries Distributed N(0,1)")

# Plot histogram
plt.hist(evals, density = True, bins = 20)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.plot(x, norm.pdf(x, 27.7, 0.5), "k", linewidth = 3, color = "black")

#%%
### Problem 0(C): Generate 1,000 200x200 symmetric matrices whose entries 
###               are independent N(0,1) random variables. Compute the largest
###               between consecutive eigenvalues (sorted in increasing order)
###               and plot the histogram. Guess the distribution.

# Initialize empty array of largest gaps between consecutive eigenvalues
max_gaps = np.array([])

# Generate 1000 200x200 matrices with N(0,1) entries and save the largest gap
# between consecutive eigenvalues for each.
for i in range(1000):
    A = np.random.randn(200, 200)
    # Make randomly-generated matrix A symmetric
    for j in range(200):
        for k in range(j+1, 200):
            A[j, k] = A[k, j]
    # Calculate eigenvalues and sorts ascending
    lambdas = sla.eigh(A, eigvals_only = True)
    lambdas = np.sort(lambdas)
    # Generates an empty list to store gaps between consecutive eigenvalues
    gaps = np.array([])
    # Populates our list with consecutive eigenvalue gaps
    for j in range(199):
        gaps = np.append(gaps, lambdas[j+1] - lambdas[j])
    # Adds max gap between consecutive eigenvalues to our overall/running list
    max_gaps = np.append(max_gaps, np.max(gaps))

## Histogram
# Labels
plt.xlabel("Maximum Gap Between Consecutive Eigenvalues")
plt.ylabel("Frequency")
plt.title("Histogram of Maximum Gaps Between Consecutive Eigenvalues\
          \nof 200 x 200 Matrices with Entries Distributed N(0,1)")

# Plot histogram
plt.hist(max_gaps, density = True, bins = 20)
plt.show()

#%%
### Problem 0(D): Investigate the behavior of the singular values of
###               symmetric n x n random matrices for n = 200, 400, 800,
###               1600, using 100 trials for each. Plot histograms.
   
#%%
# 200 x 200 Symmetric Matrices
singular_vals_200 = singular_vals_simulation(200)[0]
graph_hist_singular_vals(singular_vals_200)

#Label
text_kwargs = dict(ha='center', va='top')
center = np.histogram(singular_vals_200, density = True, bins = 20)[1][20//2]
height = np.max(np.histogram(singular_vals_200, density = True, bins = 20)[0]) / 2
plt.text(center, height, "n = 200", text_kwargs)

plt.show()

#%%
# 400 x 400 Symmetric Matrices
singular_vals_400 = singular_vals_simulation(400)[0]
graph_hist_singular_vals(singular_vals_400)

# Label
text_kwargs = dict(ha='center', va='top')
center = np.histogram(singular_vals_400, density = True, bins = 20)[1][20//2]
height = np.max(np.histogram(singular_vals_400, density = True, bins = 20)[0]) / 2
plt.text(center, height, "n = 400", text_kwargs)

plt.show()

#%%
# 800 x 800 Symmetric Matrices
singular_vals_800 = singular_vals_simulation(800)[0]
graph_hist_singular_vals(singular_vals_800)

# Label
text_kwargs = dict(ha='center', va='top')
center = np.histogram(singular_vals_800, density = True, bins = 20)[1][20//2]
height = np.max(np.histogram(singular_vals_800, density = True, bins = 20)[0]) / 2
plt.text(center, height, "n = 800", text_kwargs)

plt.show()

#%%
# 1600 x 1600 Symmetric Matrices
singular_vals_1600 = singular_vals_simulation(1600)[0]
graph_hist_singular_vals(singular_vals_1600)

# Label
text_kwargs = dict(ha='center', va='top')
center = np.histogram(singular_vals_1600, density = True, bins = 20)[1][20//2]
height = np.max(np.histogram(singular_vals_1600, density = True, bins = 20)[0]) / 2
plt.text(center, height, "n = 1600", text_kwargs)

plt.show()

#%%
### Problem 0(E): Investigate the behavior of the condition numbers of
###               symmetric n x n random matrices for n = 200, 400, 800,
###               1600, using 100 trials for each. Plot histograms.

# We'll continue to use our singular_values_simulation(n) function
# from Problem 0(D).

#%%
# 200 x 200 Symmetric Matrices
condition_numbers_200 = singular_vals_simulation(200)[1]
graph_hist_condition_numbers(condition_numbers_200)
plt.title("Histogram of Condition Numbers of 200 x 200 Matrices\
              \nwith Entries Distributed N(0,1)")
plt.show()

#%%
# 400 x 400 Symmetric Matrices
condition_numbers_400 = singular_vals_simulation(400)[1]
graph_hist_condition_numbers(condition_numbers_400)
plt.title("Histogram of Condition Numbers of 400 x 400 Matrices\
              \nwith Entries Distributed N(0,1)")
plt.show()

#%%
# 800 x 800 Symmetric Matrices
condition_numbers_800 = singular_vals_simulation(800)[1]
graph_hist_condition_numbers(condition_numbers_800)
plt.title("Histogram of Condition Numbers of 800 x 800 Matrices\
              \nwith Entries Distributed N(0,1)")
plt.show()

#%%
# 1600 x 1600 Symmetric Matrices
condition_numbers_1600 = singular_vals_simulation(1600)[1]
graph_hist_condition_numbers(condition_numbers_1600)
plt.title("Histogram of Condition Numbers of 1600 x 1600 Matrices\
              \nwith Entries Distributed N(0,1)")
plt.show()

# Problem 1

#%%
### Problem 1(A): Write a function that solves a linear system using the 
###               Cholesky decomposition.



# # TEST
# ## Generate a test SPD matrix A
# A = np.random.randn(2,2)
# A = A @ A.T
# print(A)
# ## Generate some vector b
# b = [np.e, np.pi]
# ## Solve for x and confirm A @ x = b
# x = solve_chol(A, b)
# print(A @ x)

#%%
### Problem 1(B): Compare efficiencies of computing Cholesky and LU
###               decompositions.

#Timing borrowed/adapted from HW0
if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, method):
        """
        Get runtimes for computation of Cholesky or LU decompositions of
        randomly generated SPD matrices.
        """
        ts = []
        if method == "Cholesky":
            for n in tqdm(ns):
                A = np.random.randn(n, n)
                A = A @ A.T
                t0 = time.time()
                L = sla.cholesky(A, lower = True)
                t1 = time.time()
                ts.append(t1 - t0)
            return ts
        elif method == "LU":
            for n in tqdm(ns):
                A = np.random.randn(n, n)
                A = A @ A.T
                t0 = time.time()
                P, L, U = sla.lu(A)
                t1 = time.time()
                ts.append(t1 - t0)
            return ts

    # Generate 10 log-spaced matrix sizes between 10 and 2,000
    sizes = np.logspace(start = np.log10(10), stop = np.log10(2000), num = 10)
    sizes = np.round(sizes).astype(int)

    # Calculate runtimes for Cholesky and LU decompositions
    tcholesky = get_runtimes(sizes, "Cholesky")
    tLU = get_runtimes(sizes, "LU")

    plt.plot(sizes, tcholesky, color = "red", label = "Cholesky Decomposition")
    plt.plot(sizes, tLU, color = "blue", label = "LU Decomposition")
    plt.legend(loc = "upper center")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("SPD Matrix Dimension n")
    plt.ylabel("Decomposition Runtime")
    plt.title("Comparative Runtimes of Cholesky and LU\
              \n Decompositions of Symmetric Positive Definite (\"SPD\") Matrices")
    plt.show()

#%%
### Problem 1(C): Write a function matrix_pow(A, n) which computes A^n
###               via the Eigenvalue decomposition.

### TEST
# A = np.array([[2, 1],
#               [1, 5]])
# print(A)
# print(matrix_pow(A, 10))

#%%
### Problem 1(D): Write a function abs_det(A) that computes the absolute value
###               of the determinant of a square matrix A using its LU
###               decomposition.



# ### TEST
# # A = np.random.randn(2, 2)
# # A = np.array([[1, 2, 3],
# #               [4, 5, 6],
# #               [7, 8, 8]])
# # print(A)
# # print(abs_det(A))

# Problem 2

## Problem 2(A)
#%%


# Below, we demonstrate that, using our my_complex class, the product of 
# (1 + 1i) and its complex conjugate is 2.
x = my_complex(1, 1)
x_conj = x.conj()
print(x, "*", x_conj, "=", x*x_conj)

## Problem 2(B)

#%%
#Timing borrowed/adapted from HW0
if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, cls):
        """
        Get runtimes for norm(complex_high_dim(n)).
        """
        ts = []
        if cls == "my_complex":
            for n in tqdm(ns):
                t0 = time.time()
                vec = complex_high_dim(n, "my_complex")
                norm_vec = norm(vec, "my_complex")
                t1 = time.time()
                ts.append(t1 - t0)
            return ts
        elif cls == "cdouble":
            for n in tqdm(ns):
                t0 = time.time()
                vec = complex_high_dim(n, "cdouble")
                norm_vec = norm(vec, "cdouble")
                t1 = time.time()
                ts.append(t1 - t0)
            return ts

    n_complex_vec = np.logspace(0, 7, 8).astype(int)
    t_my_complex = get_runtimes(n_complex_vec, "my_complex")
    t_cdouble = get_runtimes(n_complex_vec, "cdouble")

    ## Graph
    plt.plot(n_complex_vec, t_my_complex, color = "red", label = "Computations with my_complex Class")
    plt.plot(n_complex_vec, t_cdouble, color = "blue", label = "Computations with np.cdouble Class")
    plt.legend(loc = "upper center")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Comparative Runtimes of Norm-Computing Algorithm for Elements of C^n")
    plt.xlabel("Dimension of C^n (n)")
    plt.ylabel("Runtime")
    plt.show()