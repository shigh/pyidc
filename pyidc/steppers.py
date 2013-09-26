from scipy import optimize

def functional_iteration(f, x0, tol=10**-16):
    """ Thin wrapper of optimize.fixed_point
    """
    return optimize.fixed_point(f, x0)

def fe(f, t0, y0, h):
    """ Forward Euler
    """
    return h*f(t0, y0) + y0

def be(f, t0, y0, h):
    """ Backward Euler
    """
    g = lambda x: h*f(t0+h, x) + y0
    U = functional_iteration(g, y0)
    return U

def rk2(f, t0, y0, h):
    """ Second order explicit RK
    """
    k1 = h*f(t0, y0)
    k2 = h*f(t0+.5*h, y0 + .5*k1)
    return y0 + k2

def impmid(f, t0, y0, h):
    """ Implicit midpoint
    """
    g = lambda x: f(t0+.5*h, y0+.5*h*x)
    k1 = functional_iteration(g, y0)
    return y0 + h*k1

def lobattoIIIA(f, t0, y0, h):
    """ Implicit second order RK
    """
    k1 = f(t0, y0)
    g = lambda x: f(t0+h, y0+.5*h*(k1+x))
    k2 = functional_iteration(g, y0)
    return y0 + .5*h*(k1+k2)
