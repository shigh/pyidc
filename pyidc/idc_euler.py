import numpy as np
from collections import namedtuple
from idc_utils import *
from intmat import interp_int, integration_matrix

def make_method(base, solver):

    def make(f, ps, dtype=np.double):
        m = base(f, ps, dtype)
        m.solver = solver
        return m

    return make

class IDCBase(object):

    def __init__(self, f, ps, dtype=np.double):
        self.f  = f
        self.ps = ps
        self.dtype = dtype
        self.shape = problem_shape(ps)
        self.h  = get_h(self.ps)
        t_vals = np.linspace(ps.rng[0], ps.rng[1], ps.n_points).reshape((ps.n_points,1))
        self.t_vals = t_vals

    def get_epi(self):
        _, rng, n_points = self.ps
        f = self.f
        eta_diff = self.eta - self.eta[0]
        t_vals = self.t_vals
        f_vals = evaluate_many(f, t_vals, self.eta, self.dtype)
        int_mat = integration_matrix(n_points)
        epi = self.h*int_mat.dot(f_vals)
        
        return eta_diff - epi

    def predict(self):
        y0, rng, n_points = self.ps
        t0=rng[0]; h=self.h; f=self.f
        
        eta = np.zeros(self.shape, dtype=self.dtype)
        eta[0] = y0
        for i in range(1, n_points):
            tp = t0 + h*(i-1)
            eta[i] = self.one_step(f, tp, eta[i-1], h)
            
        self.eta = eta
        

class IDCSingle(IDCBase):

    def correct(self):
        y0, rng, n_points = self.ps
        t0=rng[0]; h=self.h; f=self.f
        
        eta = self.eta.copy()
        epi = self.get_epi()
        t_vals = self.t_vals.flatten()
        etai = InterpFunc(t_vals, eta)
        epii = InterpFunc(t_vals, epi)
        g = lambda t, x: f(t, etai(t)+x)-f(t, etai(t)) -(epi[i]-epi[i-1])/h
        
        y_vals = np.zeros(self.shape, dtype=self.dtype)
        for i in range(1, n_points):
            tp = h*(i-1) + t0
            y_vals[i] = self.one_step(g, tp, y_vals[i-1], h)

        self.eta = eta + y_vals

    def one_step(self, f, t0, y0, h):
        return self.solver(f, t0, y0, h)


class IDCSplit(IDCBase):        

    def correct(self):
        y0, rng, n_points = self.ps
        t0=rng[0]; h=self.h; f=self.f

        eta = self.eta.copy()
        epi = self.get_epi()
        t_vals = self.t_vals.flatten()
        etai = InterpFunc(t_vals, eta)
        epii = InterpFunc(t_vals, epi)
        g1 = lambda t, x: f[0](t, etai(t)+x-epii(t))-f[0](t, etai(t))
        g2 = lambda t, x: f[1](t, etai(t)+x-epii(t))-f[1](t, etai(t))

        Q = np.zeros(self.shape, dtype=self.dtype)
        for i in range(1, n_points):
            tp = h*(i-1) + t0
            Q[i] = self.one_step([g1, g2], tp, Q[i-1], h)
            
        delta = Q - epi
        self.eta = eta + delta

    
class IDCLT(IDCSplit):

    def one_step(self, f, t0, y0, h):
        U = self.solver(f[0], t0, y0, h)
        U = self.solver(f[1], t0, U, h)
        return U


class IDCSTR(IDCSplit):
    
    def one_step(self, f, t0, y0, h):
        U = self.solver(f[0], t0, y0, h/2.)
        U = self.solver(f[1], t0, U, h)
        U = self.solver(f[0], t0+h/2., U, h/2.)
        return U

        
IDCFE = make_method(IDCSingle, fe)
IDCBE = make_method(IDCSingle, be)

IDCFELT = make_method(IDCSTR, fe)
IDCBELT = make_method(IDCSTR, be)

IDCFESTR = make_method(IDCSTR, rk2)
IDCBESTR = make_method(IDCSTR, lobattoIIIA)
