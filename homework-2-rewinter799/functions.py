"""
A library of functions
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    def plot(self, vals=np.linspace(-1,1,101), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        y_vals = self.evaluate(vals)
        plt.plot(vals, y_vals, **kwargs)
        plt.show()

    def nth_derivative(self, n):
        """
        Recursively compute the nth derivative of the function.
        """
        if n == 0:
            return self
        else:
            return self.derivative().nth_derivative(n-1)

    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        taylor_poly = Constant(self.evaluate(x0))
        
        for n in range(1, deg+1):
            nth_derivative_at_x0 = self.nth_derivative(n).evaluate(x0)
            coefficient = nth_derivative_at_x0 / math.factorial(n)
            taylor_term = Polynomial(*[0] * (n-1) + [coefficient])  # Creates a polynomial x^n with coefficient
            taylor_poly += Product(taylor_term, Power(n))  # Adds the term (x-x0)^n to the Taylor polynomial
            
        return taylor_poly


class Compose(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def evaluate(self, x):
        return self.f.evaluate(self.g.evaluate(x))

    def derivative(self):
        # Using chain rule: f'(g(x)) * g'(x)
        return Product(self.f.derivative()(self.g), self.g.derivative())

    def __str__(self):
        return self.f.__str__().format(self.g.__str__().format("{0}"))

    def __repr__(self):
        return "Compose({}, {})".format(repr(self.f), repr(self.g))


class Sum(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def evaluate(self, x):
        return self.f.evaluate(x) + self.g.evaluate(x)

    def derivative(self):
        # Sum rule: f'(x) + g'(x)
        return Sum(self.f.derivative(), self.g.derivative())

    def __str__(self):
        return "({} + {})".format(self.f.__str__().format("{0}"), self.g.__str__().format("{0}"))

    def __repr__(self):
        return "Sum({}, {})".format(repr(self.f), repr(self.g))


class Product(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def evaluate(self, x):
        return self.f.evaluate(x) * self.g.evaluate(x)

    def derivative(self):
        # Using product rule: f'(x) * g(x) + f(x) * g'(x)
        return Sum(Product(self.f.derivative(), self.g), Product(self.f, self.g.derivative()))

    def __str__(self):
        return "(({}) * ({}))".format(self.f.__str__().format("{0}"), self.g.__str__().format("{0}"))

    def __repr__(self):
        return "Product({}, {})".format(repr(self.f), repr(self.g))


class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)


class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)

class Scale(Polynomial):
    def __init__(self, a):
        # a * x + 0, which means the polynomial coefficients are [a, 0]
        super().__init__(a, 0)

class Constant(Polynomial):
    def __init__(self, c):
        # just a constant, so the polynomial coefficient is just [c]
        super().__init__(c)

class Power(AbstractFunction):
    """
    Power functions x^n
    """
    def __init__(self, n):
        """
        Power(n) creates a function x^n
        """
        self.exponent = n

    def __str__(self):
        """
        String format is ({0})^n
        """
        return f"({{0}})^{self.exponent}"

    def __repr__(self):
        """
        Official object representation is Power(n)
        """
        return f"Power({self.exponent})"

    def derivative(self):
        """
        Power Rule:
        f(x) = x^n --> f'(x) = n * x^(n-1)
        """
        return Product(Constant(self.exponent), Power(self.exponent - 1))

    def evaluate(self, x_0):
        """
        Evaluate x^n at specific value x = x_0
        """
        try:
            return x_0 ** self.exponent
        except:
            return np.nan
        
class Log(AbstractFunction):
    """
    Natural log function log(x)
    """
    def __init__(self):
        super().__init__
        
    def __str__(self):
        """
        String format is log({0})
        """
        return f"log({{0}})"
    
    def __repr__(self):
        """
        Official object representation is Log()
        """
        return f"Log()"
    
    def derivative(self):
        """
        Derivative of log(x) = 1/x
        """
        return Power(-1)
    
    def evaluate(self, x_0):
        """
        Evaluate log(x) at specific value x = x_0
        """
        try:
            return np.log(x_0)
        except:
            return np.nan
        
class Exponential(AbstractFunction):
    """
    Exponential function e^x
    """
    def __init__(self):
        super().__init__

    def __str__(self):
        """
        String format is exp({0})
        """
        return f"exp({{0}})"
    
    def __repr__(self):
        """
        Official object representation is Exponential()
        """
        return f"Exponential()"
    
    def derivative(self):
        """
        Derivative of e^x is itself: e^x
        """
        return Exponential()
    
    def evaluate(self, x_0):
        """
        Evaluate e^x at specific value x = x_0
        """
        return np.exp(x_0)
    
class Sin(AbstractFunction):
    """
    Sine function sin(x)
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        """
        String format is sin({0})
        """
        return f"sin({{0}})"
    
    def __repr__(self):
        """
        Official object representation is Sin()
        """
        return f"Sin()"
    
    def derivative(self):
        """
        Derivative of sin(x) = cos(x)
        """
        return Cos()
    
    def evaluate(self, x_0):
        """
        Evaluate sin(x) at specific value x = x_0
        """
        return np.sin(x_0)
    
class Cos(AbstractFunction):
    """
    Cosine function cos(x)
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        """
        String format is cos({0})
        """
        return f"cos({{0}})"
    
    def __repr__(self):
        """
        Official object representation is Cos()
        """
        return f"Cos()"
    
    def derivative(self):
        """
        Derivative of cos(x) = -sin(x)
        """
        return Product(Constant(-1), Sin())
    
    def evaluate(self, x_0):
        """
        Evaluate cos(x) at specific value x = x_0
        """
        return np.cos(x_0)
    
class Symbolic(AbstractFunction):
    """
    Symbolic functions represented by a string
    E.g., a function "f"
    """
    def __init__(self, f):
        self.f = f

    def __str__(self):
        """
        String format is f({0})
        """
        return f"{self.f}({{0}})"
    
    def __repr__(self):
        """
        Official object representation is Symbolic(f)
        """
        return f"Symbolic({self.f})"
    
    def derivative(self):
        """
        Derivative of f is expressed as f'

        Assumes symbolic function f is differentiable
        """
        deriv = f"{self.f}'"
        return Symbolic(deriv)
    
    def evaluate(self, x_0):
        """
        Symbolic evaluation of f at x = x_0.
        E.g., Symbolic.evaluate(x_0) is represented as f(x_0)
        """
        return f"{self.f}({x_0})"


# Problem 1 
def newton_root(f, x0, tol=1e-8):
    if not isinstance(f, AbstractFunction):
        raise TypeError("not AbstractFunction.")

    x = x0
    while abs(f(x)) >= tol:
        derivative_f = f.derivative()
        x = x - f(x) / derivative_f(x)
    return x

def newton_extremum(f, x0, tol=1e-8):
    x = x0
    while abs(f.derivative().evaluate(x)) > tol:
        # Newton's update formula
        x = x - f.derivative().evaluate(x) / f.derivative().derivative().evaluate(x)
    return x
