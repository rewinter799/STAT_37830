"""
fibonacci

functions to compute fibonacci numbers

Complete problems 2 and 3 in this file.
"""

import time # to compute runtimes
from tqdm import tqdm # progress bar

import numpy as np
import matplotlib.pyplot as plt

# Question 2
def fibonacci_recursive(n):
    """
    Returns the n-th Fibonacci number using a recursive algorithm.

    Argument n >= 0 is the index of the Fibonacci number to be returned.
    """
    if n == 0:
        # Base case for 0th Fibonacci number
        return 0
    elif n == 1:
        # Base case for 1st Fibonacci number
        return 1
    else:
        # Returns the n-th Fibonacci number by its definition
        return fibonacci_recursive(n-2) + fibonacci_recursive(n-1)


# Question 2
def fibonacci_iter(n):
    """
    Returns the n-th Fibonacci number using an iterative algorithm.

    Argument n >= 0 is the index of the Fibonacci number to be returned.
    """
    # Base Cases: 0th and 1st Fibonacci numbers
    a = 0
    b = 1
    if n == 0:
        # Special case when n = 0, as range(n-1) is empty
        return 0
    else:
        # Iteratively updates a and b to the next two Fibonacci numbers
        for i in range(n-1):
            a, b = b, a+b
        return b

print("We compare the Fibonacci numbers as produced by our recursive and \
      \niterative algorithms.")
print("Recursive \t Iterative")
for i in range(30):
    print(fibonacci_recursive(i), "\t\t", fibonacci_iter(i))
print("\n")

# Question 3
def matrix_power(A, n):
    """
    Computes the n-th power of the 2x2 matrix A, using an algorithm inspired by
    the Egyptian multiplication algorithm.

    Argument A is a 2x2 matrix.
    Argument n >= 0 is an integer indicating the power to which A will be raised.
    """
    if n == 0:
        # Base Case: Any matrix to the 0th power is the identity matrix
        return np.array([[1, 0],
                         [0, 1]])
    if n == 1:
        # Base Case: Any matrix to the 1st power is itself
        return A
    
    if n % 2 == 1:
        # Odd n: Take the square of A, with an extra multiplication at the end
        return np.matmul(matrix_power(np.matmul(A, A), n // 2), A)
    else:
        # Even n: Take the square of A
        return matrix_power(np.matmul(A, A), n // 2)
    
def fibonacci_power(n):
    """
    Returns the n-th Fibonacci number by computing the (n-1)th power of the
    Fibonacci number-generating matrix A = [[1, 1], [1, 0]], and multiplying
    this by a vector containing the 1st and 0th Fibonacci numbers: [1, 0].

    Argument n >= 0 is the index of the Fibonacci number to be returned.
    """
    # Base Case: 0th Fibonacci number is 0
    if n == 0:
        return 0
    
    # Initializes the Fibonacci number-generating matrix
    A = np.array([[1, 1],
                  [1, 0]])
    
    # Calculates the (n-1)th power of the Fibonacci number-generating matrix,
    # to compute the n-th Fibonacci number
    A_n_minus_one = matrix_power(A, n-1)

    # Initializes a vector with the 1st and 0th Fibonacci numbers (1 and 0)
    x_1 = np.array([1, 0])

    # Computes the n-th and (n-1)th Fibonacci numbers, returning the n-th
    x_n = A_n_minus_one @ x_1
    return x_n[0]

print("We now compare the Fibonacci numbers as produced by our recursive, \
      \niterative, and matrix-vector product algorithms.")
print("Recursive \t Iterative \t Matrix-Vector Product")
for i in range(26):
    print(fibonacci_recursive(i), "\t\t",
          fibonacci_iter(i), "\t\t", 
          fibonacci_power(i))
for i in range(26, 30):
    print(fibonacci_recursive(i), "\t\t",
          fibonacci_iter(i), "\t", 
          fibonacci_power(i))
print("\n")

if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, f):
        """
        get runtimes for fibonacci(n)

        e.g.
        trecursive = get_runtimes(range(30), fibonacci_recusive)
        will get the time to compute each fibonacci number up to 29
        using fibonacci_recursive
        """
        ts = []
        for n in tqdm(ns):
            t0 = time.time()
            fn = f(n)
            t1 = time.time()
            ts.append(t1 - t0)

        return ts


    nrecursive = range(35)
    trecursive = get_runtimes(nrecursive, fibonacci_recursive)

    niter = range(10000)
    titer = get_runtimes(niter, fibonacci_iter)

    npower = range(10000)
    tpower = get_runtimes(npower, fibonacci_power)

    ## write your code for problem 4 below...

    plt.plot(nrecursive, trecursive, color = "red", label = "Recursive Algorithm")
    plt.plot(niter, titer, color = "blue", label = "Iterative Altorithm")
    plt.plot(npower, tpower, color = "green", label = "Matrix-Vector Product Algorithm")
    plt.legend(loc = "upper right")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Comparative Runtimes of Fibonacci Number-Generating Algorithms")
    plt.xlabel("n-th Fibonacci Number")
    plt.ylabel("Algorithm Runtime")
    plt.show()