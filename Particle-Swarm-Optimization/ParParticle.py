import numpy as np
class Particle:
    def __init__(self, dims, search_domain, test_function):
        self.fitness_function = test_function
        self.current_position = np.random.uniform(low=search_domain[0], high=search_domain[1], size=dims) 
        self.current_value = self.fitness_function.evaluate(self.current_position)
        self.pbest_position = self.current_position.copy()
        self.pbest_value = self.current_value.copy()
        self.velocity = np.random.uniform(low=search_domain[0], high=search_domain[1], size=dims)

    def update_pbest(self, position):
        self.current_position = position
        self.current_value = self.fitness_function.evaluate(position)
        if self.current_value < self.pbest_value:
            self.pbest_value = self.current_value
            self.pbest_position = self.current_position
   
    def move(self, velocity):
        self.current_position = self.current_position + velocity
