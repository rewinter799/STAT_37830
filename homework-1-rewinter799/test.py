import unittest

from matlib import *
import numpy as np

tol = 1e-8

class TestChol(unittest.TestCase):

	def setUp(self):
		pass

	def test_solve_chol(self):
		m = 10
		n = 20
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)
			A = A @ A.T
			x = np.random.rand(m)
			b = A @ x
			x2 = solve_chol(A, b)
			self.assertTrue(np.all(np.abs(x - x2) < tol))


class TestPow(unittest.TestCase):

	def setUp(self):
		pass

	def test_matrix_pow(self):
		m = 10
		n = 10
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)
			A = A + A.T

			An = matrix_pow(A, i)
			An2 = np.linalg.matrix_power(A, i)

			self.assertTrue(np.all(np.abs(An - An2) < tol))


class TestDet(unittest.TestCase):

	def setUp(self):
		pass

	def test_det(self):
		m = 10
		n = 10
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)

			d = abs_det(A)
			d2 = la.det(A)

			self.assertAlmostEqual(d, abs(d2))

class TestComplex(unittest.TestCase):
    def setUp(self):
        pass
    def test_complex_add(self):
            x = my_complex(1,1)
            y = my_complex(1,0)+my_complex(0,1)
            z = x*y.conj()
            self.assertAlmostEqual((z.real(),z.imag()),(2,0))

