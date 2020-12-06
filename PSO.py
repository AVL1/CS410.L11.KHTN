import numpy as np
from Particle import Particle
class PSO:
    w = 0.7298 # Inertia weight
    c1 = 1.49618
    c2 = 1.49618

    def __init__(self, dims, pop_size, gens, test_function, neighborhood_topology):
        self.particles = [Particle(dims, test_function.search_domain, test_function) for _ in range(pop_size)]
        self.gens = gens
        self.gbest_position = np.random.uniform(low=test_function.search_domain[0], high=test_function.search_domain[1], size=dims)
        self.gbest_value = float('inf')
        self.fitness_function = test_function
        self.neighborhood_topology = neighborhood_topology

    def update_pbests(self):
        # Evaluate each particle current value
        # Then update each particle 's pbest value and pbest position
        if self.neighborhood_topology == 'STAR':
            ''' 
            1 center node connects to other nodes
            '''

            pbest_value = float('inf')
            pbest_position = None
            for par in self.particles:
                par_value = self.fitness_function.evaluate(par.current_position)
                if par_value < pbest_value:
                    pbest_value = par_value
                    pbest_position = par.current_position
            for par in self.particles:
                par.update_pbest(pbest_position)

        elif self.neighborhood_topology == 'RING':
            ''' 
             1 node connects to 2 other nodes
            '''

            for i in range(len(self.particles)):
                # Get 2 neighbor nodes
                n_particles = len(self.particles)
                n1_index = (i + 1) % n_particles
                n2_index = (i + n_particles - 1) % n_particles

                # Get values of 3 particles
                current_particle_value = self.fitness_function.evaluate(self.particles[i].current_position)
                n1_particle_value = self.fitness_function.evaluate(self.particles[n1_index].current_position)
                n2_particle_value = self.fitness_function.evaluate(self.particles[n2_index].current_position)
                best_value_of_3 = max(current_particle_value, n1_particle_value, n2_particle_value)

                # Update pbest of 3 particles
                for ind in (i, n1_index, n2_index):
                    if self.particles[ind].current_value == best_value_of_3:
                        self.particles[i].update_pbest(self.particles[ind].current_position)
                        self.particles[n1_index].update_pbest(self.particles[ind].current_position)
                        self.particles[n2_index].update_pbest(self.particles[ind].current_position)
                        break
        
        else:
            print('Neighborhood Topology required.')

    def update_gbest(self):
        # Update global best particle
        for par in self.particles:
            current_particle_value = par.pbest_value
            if current_particle_value < self.gbest_value:
                self.gbest_position = par.pbest_position
                self.gbest_value = par.pbest_value

    def move_particles(self):
        for par in self.particles:
            new_velocity = self.w * par.velocity \
                         + self.c1 * (par.pbest_position - par.current_position) \
                         + self.c2 * (self.gbest_position - par.current_position)

            par.move(new_velocity)

    def run(self):
        for i in range(self.gens):
            #1. Evaluate the fitness of each particle 
            #   and update individual best fitnesses and positions
            self.update_pbests()

            #2. Update global best fitness and position
            self.update_gbest()

            #3. Update velocity and position of each particle
            self.move_particles()
        # Check terminations
        # End after 50 gens
        print('Best position:', self.gbest_position)
        print('Best fitness:', self.gbest_value)
        print('Global minimum:', self.fitness_function.global_minimum)
        print('Distance:', abs(self.gbest_value - self.fitness_function.global_minimum))