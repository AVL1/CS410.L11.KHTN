from PSO import PSO
from TestFunctions import *

rastrigin = Rastrigin(dims=2)
rosenbrock = Rosenbrock(dims=2)
eggholder = Eggholder(dims=2)
ackley = Ackley(dims=2)

PSO = PSO(dims=2, pop_size=32, gens=50, test_function=ackley, neighborhood_topology='STAR')
PSO.run()
