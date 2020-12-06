import numpy as np
class TestFunction:
    def __init__(self, dims, global_minimum, search_domain):
        self.global_minimum = global_minimum
        self.search_domain = search_domain
        self.dims = dims

class Rastrigin(TestFunction):
    A = 10
    def __init__(self, dims):
        global_minimum = 0
        search_domain = (-5.12, 5.12)
        super(Rastrigin, self).__init__(dims, global_minimum, search_domain)

    def evaluate(self, X):
        return self.A * len(X) + sum((x**2 - self.A * np.cos(2 * np.pi * x)) for x in X)

class Rosenbrock(TestFunction):
    def __init__(self, dims):
        global_minimum = 0
        search_domain = (-9999, 9999) # (-inf, inf)
        super(Rosenbrock, self).__init__(dims, global_minimum, search_domain)
    
    def evaluate(self, X):
        return sum(100 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)

class Eggholder(TestFunction):
    def __init__(self, dims):
        global_minimum = -959.6407
        search_domain = (-512, 512)
        super(Eggholder, self).__init__(dims, global_minimum, search_domain)
    
    def evaluate(self, X):
        return -(X[1] + 47) * np.sin(np.sqrt(abs(X[0]/2 + (X[1]  + 47)))) - X[0] * np.sin(np.sqrt(abs(X[0] - (X[1]  + 47))))

class Ackley(TestFunction):
    def __init__(self, dims):
        global_minimum = 0
        search_domain = (-5, 5)
        super(Ackley, self).__init__(dims, global_minimum, search_domain)
    
    def evaluate(self, X):
        part_1 = np.exp(-0.2 * np.sqrt(0.5 * (X[0]**2 + X[1]**2)))
        part_2 = np.exp(0.5 * (np.cos(2 * np.pi * X[0]) + np.cos(2 * np.pi * X[1])))
        return -20 * part_1 - part_2 + np.exp(1) + 20
        
