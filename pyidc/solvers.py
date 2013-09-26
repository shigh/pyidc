from utils import *

def idc_solve_single(f, ps, method, order, dtype=np.double, n_corr=None):
    """ A single time step of IDC
    """
    
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
    """ Solve IDC over a full region
    """
    if isinstance(ps, list):
        out = []
        for s in ps:
            out.append(idc_solve_single(f, s, method, order, dtype, n_corr))
    else:
        out = idc_solve_single(f, ps, method, order, dtype, n_corr)

    return out

def idc_order(f, ps_base, method, order, actual, pows, ret_errs=False, dtype=np.double, n_corr=None):
    """ Estimate order of IDC method
    """
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
