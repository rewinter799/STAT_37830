"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

## Problem 0

class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Subclass of scipy's OdeSolver class, implementing the Forward Euler method.
    """
    def __init__(self, fun, t0, y0, t_bound, vectorized, h0 = 0,
                 support_complex = False, **extraneous):
        """
        Instantiates a ForwardEuler object by invoking the __init__ function 
        of the OdeSolver superclass and defining additional class attributes
        specific to the Forward Euler procedure.
        """
        
        # Warning for extaneous kwargs
        if len(extraneous) == 0:
            warn("Warning: Extraneous arguments are irrelevant and will not be used.")

        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound,
                                           vectorized, support_complex)

        self.y_old = None

        # Direction for Forward Euler method is always positive
        self.direction = +1

        # Jacobian not used in Forward Euler method
        self.njev = 0
        self.nlu = 0

        # Default step size is one one-hundredth of the time between
        # the initial time and the boundary time.
        if h0 != 0:
            self.h = h0
        else:
            self.h = (t_bound - t0) / 100

    def _step_impl(self):
        """
        Propagates FowardEuler one step further.
        
        Returns tuple (success, message).

        success is a Boolean indicating whether a step was successful.

        message is a string containing description of a failure if a step
        failed, or None otherwise.
        """
        success = True
        message = None

        self.y_old = self.y

        self.t = self.t + self.h
        self.y = self.t + self.h * self.fun(self.t, self.y)
        
        return (success, message)

    def _dense_output_impl(self):
        """
        Returns a `DenseOutput` object covering the last successful step.
        Specifically, returns a `ForwardEulerOutput` object.
        """
        return ForwardEulerOutput(self.t_old, self.t, self.y_old, self.y)
    

class ForwardEulerOutput(DenseOutput):
    """
    Subclass of scipy's DenseOutput class, which produces DenseOutput objects
    compatible with the Forward Euler procedure.
    """
    def __init__(self, t_old, t, y_old, y):
        """
        Instantiates an object of the ForwardEulerOutput class.
        
        Determines a range of time values and output values corresponding
        to the most recent step of the Forward Euler procedure.

        Sets an attribute interp, a linear spline interpolant based on the
        range of time and output values in the most recent step of the
        Forward Euler procedure.
        """
        super(ForwardEulerOutput, self).__init__(t_old, t)

        t_range = np.array([t_old, t])
        y_range = np.array([y_old[0], y[0]])

        self.interp = interp1d(t_range, y_range, kind = "linear")

    def _call_impl(self, t):
        """
        Returns the ForwardEulerOutput object's interpolant, evaluated at
        an array of time values t.
        """
        return self.interp(t)
