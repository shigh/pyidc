
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import *
from methods import *

#M = IDCBESTR
M = make_method(IDCSTR, be)
t_rng = (0, 1.)
n_points = 10
h = (t_rng[1] - t_rng[0])/float(n_points-1)
a_range = np.linspace(-1500, 20, 200)/3
b_range = np.linspace(-800, 800, 200)/3

l_vals = [[complex(a, b)/2 for a in a_range] for b in b_range]
l_vals = np.array(l_vals).flatten()
ps = ProblemSetup(np.ones(len(l_vals)).astype(np.complex), 
                  t_rng, n_points)

z_vals = l_vals
re = np.real(z_vals); im = np.imag(z_vals)

f = SplitFunction(lambda t, y: l_vals*y, lambda t, y: l_vals*y)
m = M(f, ps, np.complex)
m.predict()
ev  = []
for i in range(3):
    m.correct()
    ev += [m.eta.copy()]
    
stab = [np.abs(eta[-1]) <= 1 for eta in ev]

colors = ['b', 'r', 'g']
for i in range(len(stab)):
    plt.scatter(re[stab[i]], im[stab[i]], marker='x', color=colors[i])

plt.show()    


