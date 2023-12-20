# Homework 1 - Linear Algebra and a Basic Class

Please put any written answers in [`answers.md`](answers.md)

Reminder: you can embed images in markdown.  If you are asked to make a plot, save it as a `png` file, commit it to git, and embed it in this file.

Please complete the code [`script.ipynb`](script.ipynb) to finish all tasks.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

## Important Information

### Due Date
This assignment is due Thursday, October 12 at 10PM Chicago time.

### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Problem 0 |  |   |  | /50 |
| Part A |  | /8 | /2 | /10 |
| Part B |  | /8 | /2 | /10 |
| Part C |  | /8 | /2 | /10 |
| Part D |  | /8 | /2 | /10 |
| Part E |  | /8 | /2 | /10 |
| Problem 1 |  |   |  | /25 |
| Part A | /2 | /2 | /1 | /5 |
| Part B |  | /3 | /2 | /5 |
| Part C | /3 | /5 | /2 | /10 |
| Part D | /1 | /3 | /1 | /5 |
| Problem 2 |  |   |  | /25 |
| Part A | /8 |  | /7 | /15 |
| Part B |  | /8 | /2 | /10 |


Correctness will be based on code (i.e. did you provide was was aksed for) and the content of [`answers.md`](answers.md).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

You can run tests for the functions you will write in problems 1 and 2 using either `unittest` or `pytest` (you may need to `conda install pytest`).  From the command line:
```
python -m unittest test.py
```
or
```
pytest test.py
```
The tests are in [`test.py`](test.py).  You do not need to modify (or understand) this code.

You need to pass all tests to receive full points. Note that there are no tests for problem 0.

Please enable GitHub Actions on your repository (if it isn't already) - this will cause the autograder to run automatically every time you push a commit to GitHub, and you can get quick feedback. However, GitHub actions has a time limit associated. If running pytest and script.py takes more than 30 minutes, it will time out. Make sure to follow the instructions in the assignment to use numba appropriately so that your code runs in a reasonable amount of time.


## Problem 0 - Scipy linear algebra (50 points)

In this question the goal is to gain some familiarity with scipy's linear algebra routines by doing some experiments on random matrices.

### Part A (10 points)
Construct one hundred `1000x1000` **symmetric** matrices whose entries are independent standard normal random variables (note that for your matrices, `A[i,j] = A[j,i]`). Using scipy, compute the eigenvalues of each of these matrices and plot a histogram of **all** of the eigenvalues of all `100` matrices. In other words, you should obtain `100,000` eigenvalues in total from your `100` random matrices. Try to guess what the true distribution of eigenvalues is by looking at the histogram. Plot the error in the fit of your guess and the empirical distribution obtained from the histogram. 

Now redo the experiments with `200x200`, `400x400`, `800x800`, and `1600x1600` matrices. How do any parameters of your model scale with the size of the matrices?

Hint: Do not create all the matrices and then process them. Use a loop to sample a matrix and find its eigenvalues. Do not store all the matrices (just the eigenvalues!).

For the histogram plots: look at the documentation for matplotlib.pyplot.hist, ie. in ipython, type

`import matplotlib.pyplot as plt`

`plt.hist()`

In your response, include the code you used to run your experiments, plots of the resulting histograms, model, and fits. In your write up include your guess at the model.

### Part B (10 points)
Construct one thousand `200x200` **symmetric** matrices whose entries are independent standard normal random variables. Compute the largest eigenvalue of each matrix and plot the histogram. Can you guess what form the distribution of the largest eigenvalue takes? Include the code you ran for your experiments and a histogram of the results.

### Part C (10 points)
Construct one thousand `200x200` **symmetric** matrices whose entries are independent standard normal random variables. Plot a histogram of the largest gap between **consecutive** eigenvalues (if they are sorted in increasing order). Can you guess what form the distribution of the largest eigenvalue takes? Include the code you ran for your experiments and a histogram of the results.

### Part D (10 points)
Using scipy, investigate the behavior of the singular values of symmetric random matrices. Plot a histogram of your results for various sizes (`200,400,800,1600`) using `100` trials each.

### Part E (10 points)
Plot histograms of the condition number of random matrices (the largest singular value of a matrix divided by its smallest singular value).

## Problem 1 - Matrix Factorizations (25 points)

In this problem, you'll practice applying some matrix factorizations in `scipy.linalg`, and do some performance comparisons.  

In this problem, you'll deal with the Cholesky decomposition. First, some definitions:

A matrix is symmetric positive-definite (SPD) if
1. A is symmetric (`A = A.T`)
2. Eigenvalues of A are all non-negative (strictly greater than 0)

An easy way to generate a random SPD matrix is:
```python
A = np.random.randn(m,n) # n >= m
A = A @ A.T # use @ not + for SPD!
```

The Cholesky factorization of a SPD matrix `A` is `A = L @ L.T` where `L` is lower triangular.  Alternatively, we might say `A = U.T @ U` where `U` is `L.T`.  This is a variant of the LU decomposition, where we can use symmetry in `A`.


### Part A (5 points)

Write a function `solve_chol` which solves a linear system using the Cholesky decomposition.  Explicitly, `x = solve_chol(A, b)` should (numerically) satisfy `A @ x = b`.  You can assume that `A` is SPD.

Use [`cholesky`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html#scipy.linalg.cholesky) in `scipy.linalg` to compute the decomposition.

### Part B (5 points)

Create a log-log plot of the time it takes to compute the Cholesky decomposition vs. the LU decomposition of random n x n SPD matrices.  Compute times for 10 values of `n` logarithmically spaced between `n=10` and `n=2000`. (use `np.logspace` and `np.round`).  Make sure to give your plot axes labels, a legend, and a title.

Both factorizations take O(n^3) time to compute - which is faster in practice?

### Part C - Basic Matrix Functions (10 points)

One example of a matrix function is the matrix power `A**n`.  In Homework 0, you computed this using a version of the Egyptian algorithm.

If `A` is symmetric, recall `A` has an eigenvalue decomposition `A = Q @ L @ Q.T`, where `Q` is orthogonal, which can be computed with `eigh`.  Then we can write the power

`A**n = A @ A @ ... @ A`

as

`A**n = (Q @ L @ Q.T) @ (Q @ L @ Q.T) @ ...`

Because `Q.T @ Q = I` (the identity), this becomes

`A**n = Q @ (L**n) @ Q.T`

Recall that `L` is diagonal, so `L**n` can be computed element-wise.

Write a function `matrix_pow` where `matrix_pow(A, n)` computed via the Eigenvalue decomposition, as above.  You can assume that `A` is symmetric.  You can also just use the NumPy vectorized implementation of power for `L`.

Do you expect this function to be asymptotically faster or slower than the approach you used in Homework 0?  Considering that the Fibonacci numbers are integers, what issues might you need to consider if using this function to compute Fibonacci numbers?


### Part D - Determinant (5 points)

Calculating the determinant of a matrix is very easy using the LU decomposition.  Recall that if `A = B @ C`, that `det(A) = det(B) * det(C)`.

A useful property of lower and upper-triangular matrices `T` are that `det(T)` is just the product of the diagonal elements (you don't have to look at the off-diagonal elements).

Write a function `abs_det(A)` which computes the absolute value of the determinant of a square matrix `A` by using its LU decomposition.  One useful property of LU decompositions is that `L` by convention just has 1 on its diagonal.  Note that the determinant of a permutation matrix is either +1 or -1 depending on the parity of the permutation.

## Problem 2 - Introduction to object orientated programming - (25 pts)

### Part A (15 pts)

Write a class `my_complex` that stores a complex number. The class must contain the magic methods `__init__`, `__add__` and `__mul__`, to initialize, add, and multiply complex numbers. It must also contain a method `conj` to conjugate them and the methods `real` and `imag` that return the real and imaginary part. Demonstrate that it is working by verifying that $(1+1i)*(\overline{(1)+(1i)})$ is 2.

### Part B (10 pts)

Write a function that generates your favorite element of $\mathbb{C}^n$ using a list of instances of `my_complex`. Then write another function that computes the dot product of two vectors. Don't forget the complex conjugate.

Time how long it takes to generate the above vector and compute its norm for various values of $n$ from `1` to `1,000,000`. Using a pretty graph compare how long it takes to do the same things using `numpy` arrays of type `numpy.cdouble`. 

Based on these two experiments, how should you store vectors of complex numbers?
