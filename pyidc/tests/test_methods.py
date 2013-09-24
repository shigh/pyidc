import unittest
import numpy as np

from idc_utils import *
from idc_euler import IDCBase, IDCFE, IDCBE

class IDCTestBase(object):

    def get_f(self):
        def f(t, y):
            return t + y
        return f

    def exact(t):
        return -1 + 2*np.exp(t) - t

class IDCRealScalarBase(IDCTestBase):

    ps = ProblemSetup(1, (0, 1), 10)


    
