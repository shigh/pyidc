from utils import *

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

class IDCSolver(object):

    def __init__(self, f, ps, method, order, dtype=np.double, n_corr=None):
        self.f      = f
        self.ps     = ps
        self.method = method
        self.order  = order
        self.dtype  = dtype
        self.n_corr = n_corr

    def solve(self):
        """ A single time step of IDC
        """

        t_vals = np.linspace(self.ps.rng[0], self.ps.rng[1], self.ps.n_points)
        y_vals = np.zeros_like(t_vals).astype(self.dtype)
        y_vals[0] = self.ps.y0

        n_corr = order-1 if self.n_corr==None else self.n_corr

        for i in range(self.ps.n_points-1):
            s = ProblemSetup(y_vals[i], (t_vals[i], t_vals[i+1]), self.order)
            m = method(self.f, self.s)
            m.predict()
            for j in range(n_corr): m.correct()

            y_vals[i+1] = m.eta[-1]

        return y_vals    
        
