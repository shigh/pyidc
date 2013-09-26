import numpy as np
from scipy import interpolate, optimize
from collections import namedtuple, Iterable
from steppers import *

# Cut down on the number of constructor parameters for
# IDCBase derived methods.
ProblemSetup = namedtuple("ProblemSetup", ["y0", "rng", "n_points"])

def one_d_poisson(n, h, include_boundary=True):
    """ Dense solution matrix for 1D poisson equations

    You likely want to set h = dt/dx**2
    """
    a = np.zeros((n,n))
    if include_boundary:
        np.fill_diagonal(a[1:-1,1:-1], -2.)
        np.fill_diagonal(a[1:-1,:-1], 1.)
        np.fill_diagonal(a[1:-1,2:], 1.)
        a = a/h**2
        a[0,0]=a[-1,-1]=1.
        return  a
    else:
        np.fill_diagonal(a, -2.)
        np.fill_diagonal(a[:-1,1:], 1.)
        np.fill_diagonal(a[1:,:-1], 1.)
        a = a/h**2
        return  a

def len2(x):
    """ Handle the case where x is not iterable
    """
    if hasattr(x, "__len__"):
        return len(x)
    else:
        return 1

def problem_shape(ps):
    """ Needed to handle non-array and array inputs
    """
    return (ps.n_points, len2(ps.y0))

def get_h(ps):
    """ Constant step size
    """
    _, rng, n_points = ps 
    return (rng[1] - rng[0])/float(n_points-1)

def map_f(f, ps, dtype=np.double):
    """ Apply function over the range specified in ps
    """
    _, rng, n_points = ps
    t0 = rng[0]
    h = get_h(ps)
    
    out = np.zeros((n_points, len2(ps.y0)), dtype=dtype)
    for i in range(n_points):
        out[i] = f(t0+h*i)
    
    return out

def poly_eval_at(t_vals, t_eval, y_vals):
    """ Interpolate and evaluate
    """
    deg = len(y_vals) - 1
    p = np.polyfit(t_vals, y_vals, deg)
    return np.polyval(p, t_eval)

def evaluate_many(f, t, y, dtype=np.double):
    """ Evaluate func at many points
    """
    n = len(t)
    out = np.zeros((n, len2(y[0])), dtype=dtype)

    for i in range(n):
        out[i] = f(t[i], y[i])

    return out

class SplitFunction(object):
    """ Can behave as a list of functions or a single function
    """

    def __init__(self, *funcs):
        self.funcs = funcs

    def f(self, t, y):
        out = self.funcs[0](t, y)
        for func in self.funcs[1:]:
            out += func(t, y)
        return out

    def __call__(self, t, y):
        return self.f(t, y)

    def __getitem__(self, i):
        return self.funcs[i]

    def __len__(self):
        return len(self.funcs)

class InterpFunc(object):
    """ Simple interpolate wrapper
    """

    def __init__(self, t_vals, y_vals):
        self.deg = len(y_vals) - 1
        self.shift = np.min(t_vals)
        self.scale = 1./(np.max(t_vals) - np.min(t_vals))
        scaled_t_vals = (t_vals - self.shift)*self.scale
        self.p = np.polyfit(scaled_t_vals, y_vals, self.deg)

    def eval(self, t):
        scaled_t = (t - self.shift)*self.scale
        return np.polyval(self.p, scaled_t)

    def __call__(self, t):
        return self.eval(t)
