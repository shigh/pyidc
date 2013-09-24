import numpy as np

### Define Problems

class TestFunction(object):

    def exact(self, t):
        return 0

    def evaluate(self, t, y):
        return 0

    def exact_many(self, t):
        n = len(t)
        out = np.zeros_like(t, dtype=np.double)

        for i in range(n):
            out[i] = self.exact(t[i])

        return out

    def __call__(self, t, y):
        return self.evaluate(t, y)

class TestFunction1(TestFunction):
    """ y' = .5 y
        y(0) = 1

        y(t) = exp(.5 t)
    """
    
    def __init__(self):
        self.y0 = 1
    
    def evaluate(self, t, y):
        return .5*y
    
    def exact(self, t):
        return np.exp(.5*t)
    
class TestFunction2(TestFunction):
    """ y' = y + t
        y(0) = 1
        
        y(t) = -1 + 2 e^t - t
    """
    
    def __init__(self):
        self.y0 = 1
    
    def evaluate(self, t, y):
        return y + t
    
    def exact(self, t):
        return -1 + 2*np.exp(t) - t
    
class TestFunction3(TestFunction):
    """ y' = cos(t)
        y(0) = 0
        
        y(t) = sin(t)
    """
    
    def __init__(self):
        self.y0 = 0
        
    def evaluate(self, t, y):
        return np.cos(t)
    
    def exact(self, t):
        return np.sin(t)

###
