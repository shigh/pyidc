import numpy as np
import sympy as sym
from sympy import Symbol, Matrix, MatrixSymbol
from sympy.abc import x, h, t

def c_list(t, t_vals):
    """ List of components in the polynomial expansion
    """
    n_points = len(t_vals)
    out = []
    for j in range(n_points):
        c = 1
        rng = range(n_points)
        rng.remove(j)
        for k in rng:
            c *= (t-t_vals[k])/(t_vals[j] - t_vals[k])
        out.append(c)
        
    return out

def build_integration_matrix(n_points, as_np=True):
    """ The integral from t_0 to t_n
    """
    c_vals = c_list(t, [i*h for i in range(n_points)])
    c_vals = [c.expand() for c in c_vals]
    c_vals = [c.integrate(t) for c in c_vals]

    M = sym.zeros(n_points, n_points)
    for i in range(n_points):
        for j in range(n_points):
            M[i,j] = c_vals[j].subs(t, i*h)
    M = M/h
    
    if as_np:
        return np.array(M.evalf()).astype(np.double)
    else:
        return M

# Only calculate int mats once        
_mat_cache = {}
def integration_matrix(n_points):
    """ Use sympy to build integration matrix
    """
    if _mat_cache.has_key(n_points):
        return _mat_cache[n_points]

    M = build_integration_matrix(n_points, True)
    _mat_cache[n_points] = M

    return M

def interp_int(x_vals, y_vals):
    """ Use integration matrix to interpolate diffs
    """
    h = x_vals[1] - x_vals[0]
    M = integration_matrix(len(y_vals))
    mat_int  = h*M.dot(y_vals)
    mat_int[1:] = mat_int[1:] - mat_int[:-1]
    mat_int[0] = 0
    
    return mat_int
