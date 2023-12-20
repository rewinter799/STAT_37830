import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, BarycentricInterpolator
from scipy.special import roots_chebyt, roots_legendre
from scipy.integrate import newton_cotes
import matplotlib.pyplot as plt

## Problems 0(A), 1
class BarycentricInterval():
    """
    Barycentrically interpolates a function f on the interval I = [a, b]
    using a degree (n-1) polynomial passing through n nodes, spaced either
    Chebyshev, Legendre, or equi-spaced.

    Attributes:
    f: Function to be estimated/interpolated.
    a: Left endpoint of interval I.
    b: Right endpoint of interval I.
    n: Number of nodes through which interpolant is constructed;
       (n-1) is degree of polynomial interpolant.
    nodes: ndarray of nodes through which interpolant is constructed.
    vals: ndarray of values of the function f at each input in nodes.
    quad_weights: ndarray of weights that, when multiplied by a corresponding
                  ndarray of function values, returns an estimate of the
                  integral of f from a to b.
    """
    def __init__(self, f, a, b, n, node_type = "Chebyshev"):
        """
        Instantiates an object of the BarycentricInterval class,
        which interpolates a function f on the interval I = [a, b].

        Parameters:
        f: Function to be estimated/interpolated.
        a: Left endpoint of interval I.
        b: Right endpoint of the interval I.
        n: Number of nodes through which interpolant is constructed;
           (n-1) is degree of polynomial interpolant.
        node_type: Determines type of nodes, either Chebyshev, Gauss-
                   Legendre, or equispaced.
        """
        self.f = f
        self.a = a
        self.b = b
        self.n = n
        self.node_type = node_type

        # Create array of nodes
        if node_type == "Chebyshev":
            self.nodes = sp.special.roots_chebyt(n)[0]
            for i in range(n):
                self.nodes[i] = self.nodes[i] * (b - a)/(1 - -1) # stretch
                self.nodes[i] = self.nodes[i] + (a + b)/2        # shift
        elif node_type in np.array(["Legendre", "Gauss", "Gauss-Legendre"]):
            self.nodes = sp.special.roots_legendre(n)[0]
            for i in range(n):
                self.nodes[i] = self.nodes[i] * (b - a)/(1 - -1) # stretch
                self.nodes[i] = self.nodes[i] + (a + b)/2        # shift
        elif node_type in np.array(["Equispaced", "Equi-Spaced", "equi"]):
            self.nodes = np.linspace(a, b, n)

        # Create array of function values at nodes
        self.vals = f(self.nodes)

        # Create interpolant
        self.interp = BarycentricInterpolator(self.nodes, self.vals)

        # Create quadrature weights
        if node_type in np.array(["Legendre", "Gauss", "Gauss-Legendre"]):
            self.quad_weights = sp.special.roots_legendre(self.n)[1]
        elif node_type in np.array(["Equispaced", "Equi-Spaced", "equi"]):
            self.quad_weights = sp.integrate.newton_cotes(self.n - 1, 1)[0]

    def __call__(self, x):
        """
        Evaluates and returns interpolant at some point x.

        Parameter x is the point at which the interpolant is evaluated.
        """
        return self.interp(x)

    def __string__(self):
        """
        Returns string representation/summary of interpolant.
        """
        return f"Barycentric -- {self.node_type} Nodes"

    def quad(self):
        """
        Estimates the integral of f from a to b using n weights generated
        based on node_type.

        Returns estimate of integral.
        """
        integral = 0
        for i in range(self.n):
            integral += self.vals[i] * self.quad_weights[i]
        return integral

## Problems 0(B), 1
class LinSplineInterval():
    """
    Piecewise linearly interpolates a function f on the interval I = [a, b]
    using a degree (n-1) polynomial passing through n equally-spaced nodes.

    Attributes:
    f: Function to be estimated/interpolated.
    a: Left endpoint of interval I.
    b: Right endpoint of interval I.
    n: Number of nodes through which interpolant is constructed;
       (n-1) is degree of polynomial interpolant.
    nodes: ndarray of nodes through which interpolant is constructed.
    vals: ndarray of values of the function f at each input in nodes.
    """
    def __init__(self, f, a, b, n):
        """
        Instantiates an object of the LinSplineInterval class,
        which creates a piecewise linear interpolant of a function f 
        on the interval I = [a, b], with equally-spaced nodes.

        Parameters:
        f: Function to be estimated/interpolated.
        a: Left endpoint of interval I.
        b: Right endpoint of the interval I.
        n: Number of nodes through which interpolant is constructed;
           (n-1) is degree of polynomial interpolant.
        """
        self.f = f
        self.a = a
        self.b = b
        self.n = n

        # Create array of nodes and evaluate f at nodes
        self.nodes = np.linspace(a, b, n)
        self.vals = f(self.nodes)

        # Create interpolant
        self.interp = interp1d(self.nodes, self.vals, kind = "linear")

    def __call__(self, x):
        """
        Evaluates and returns interpolant at some point x.

        Parameter x is the point at which the interpolant is evaluated.
        """
        return self.interp(x)
    
    def __string__(self):
        """
        Returns string representation/summary of interpolant.
        """
        return f"Linear Spline -- Equi-Spaced Nodes"

    def quad(self):
        """
        Estimates the integral of f from a to b using the trapezoid rule.

        Returns estimate of integral.
        """
        integral = sp.integrate.trapezoid(y = self.vals, x = self.nodes)
        return integral

## Problem 0(C)
def plot_data(dats, legs, xlab, ylab):
    """
    Generates plots of data.

    Parameter dats is an ndarray of data.
    Parameter legs is an ndarray of legend entries, corresponding to
    each element of dats.
    Parameter xlab is the x-axis label.
    Parameter ylab is the y-axis label.
    """
    dats = np.array(dats)
    
    markers = np.array(["o", "v", "s", "*", ".", ",", "^", "<", ">", "1",
                        "2", "3", "4", "p", "h", "H", "+", "x", "D", "d",
                        "|", "_"])

    for i in range(len(dats)):
        plt.plot(dats[i][0], dats[i][1], label = legs[i],
                 linewidth = 2, alpha = 1, marker = markers[i])
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc = "best")
    plt.grid(which = "major")