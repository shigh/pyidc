import numpy as np
from scipy import interpolate, optimize
from collections import namedtuple, Iterable

ProblemSetup = namedtuple("ProblemSetup", ["y0", "rng", "n_points"])

def one_d_poisson(n, h, include_boundary=True):
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
    if hasattr(x, "__len__"):
        return len(x)
    else:
        return 1

def problem_shape(ps):
    return (ps.n_points, len2(ps.y0))

def get_h(ps):
    _, rng, n_points = ps 
    return (rng[1] - rng[0])/float(n_points-1)

def map_f(f, ps, dtype=np.double):
    _, rng, n_points = ps
    t0 = rng[0]
    h = get_h(ps)
    
    out = np.zeros((n_points, len2(ps.y0)), dtype=dtype)
    for i in range(n_points):
        out[i] = f(t0+h*i)
    
    return out

def functional_iteration(f, x0, tol=10**-16):
    return optimize.fixed_point(f, x0)

def poly_eval_at(t_vals, t_eval, y_vals):
    deg = len(y_vals) - 1
    p = np.polyfit(t_vals, y_vals, deg)
    return np.polyval(p, t_eval)

def evaluate_many(f, t, y, dtype=np.double):

    n = len(t)
    out = np.zeros((n, len2(y[0])), dtype=dtype)

    for i in range(n):
        out[i] = f(t[i], y[i])

    return out

def idc_solve_single(f, ps, method, order, dtype=np.double, n_corr=None):
    
    t_vals = np.linspace(ps.rng[0], ps.rng[1], ps.n_points)
    y_vals = np.zeros_like(t_vals).astype(dtype)
    y_vals[0] = ps.y0

    n_corr = order-1 if n_corr==None else n_corr
    
    for i in range(ps.n_points-1):
        s = ProblemSetup(y_vals[i], (t_vals[i], t_vals[i+1]), order)
        m = method(f, s)
        m.predict()
        for j in range(n_corr): m.correct()
        
        y_vals[i+1] = m.eta[-1]
    
    return y_vals    

def idc_solve(f, ps, method, order, dtype=np.double, n_corr=None):
    if isinstance(ps, list):
        out = []
        for s in ps:
            out.append(idc_solve_single(f, s, method, order, dtype, n_corr))
    else:
        out = idc_solve_single(f, ps, method, order, dtype, n_corr)

    return out

def idc_order(f, ps_base, method, order, actual, pows, ret_errs=False, dtype=np.double, n_corr=None):
    ps = [ProblemSetup(ps_base.y0, ps_base.rng, 2**k) for k in pows]
    sols = idc_solve(f, ps, method, order, dtype, n_corr)
    errs = np.log2([abs(actual - sol[-1]) for sol in sols])
    
    A = np.ones((len(pows), 2))
    A[:,0] = pows
    slope = np.linalg.lstsq(A, errs)[0][0]
    
    if ret_errs:
        return (-slope, errs)
    else:
        return -slope    

class SplitFunction(object):

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

class InterpFunc(object):

    def __init__(self, t_vals, y_vals):
        self.deg = len(y_vals) - 1
        self.p = np.polyfit(t_vals, y_vals, self.deg)

    def eval(self, t):
        return np.polyval(self.p, t)

    def __call__(self, t):
        return self.eval(t)

def fe(f, t0, y0, h):
    return h*f(t0, y0) + y0

def be(f, t0, y0, h):
    g = lambda x: h*f(t0+h, x) + y0
    U = functional_iteration(g, y0)
    return U

def rk2(f, t0, y0, h):
    k1 = h*f(t0, y0)
    k2 = h*f(t0+.5*h, y0 + .5*k1)
    return y0 + k2

def impmid(f, t0, y0, h):
    g = lambda x: f(t0+.5*h, y0+.5*h*x)
    k1 = functional_iteration(g, y0)
    return y0 + h*k1

def lobattoIIIA(f, t0, y0, h):
    k1 = f(t0, y0)
    g = lambda x: f(t0+h, y0+.5*h*(k1+x))
    k2 = functional_iteration(g, y0)
    return y0 + .5*h*(k1+k2)
    
