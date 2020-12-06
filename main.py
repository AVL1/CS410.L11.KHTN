from PSO import PSO
from TestFunctions import *

T = ['STAR', 'RING']
N = [128, 256, 512, 1024, 2048]
rastrigin10 = Rastrigin(dims=10)
rosenbrock10 = Rosenbrock(dims=10)
funcs = [(rastrigin10, 'Rastrigin10'), (rosenbrock10, 'Rosenbrock10')]
# 10 sets (T, N)
# Each set run 10 times

for t in T:
    for n in N:
        with open('log.txt', 'w') as f:
            f.write('{}({}, {})\n'.format(str(f), t, n))

        # Write final results
        with open('result_after_10_times.txt', 'w') as f2:
            f2.write('{}({}, {})\n'.format(f, t, n))

        # Run 10 times
        pso = PSO(dims=10, pop_size=n, gens=0, test_function=rastrigin10, neighborhood_topology=t) # gens is defined but not used
        random_base = 18520473
        MAX_N_EVALS = 1000000
        res = [] # contains value of each run
        for i in range(10):
            rand = random_base + i
            np.random.seed(rand)
            n = 0
            # Run PSO
            while True:
                pso.update_pbests()
                pso.update_gbest()
                pso.move_particles()
                # 1 run takes 4 times for evaluation
                n += 4
                if n >= 1000000:
                    break
            
            # Save result of each run to calculate mean
            res.append(pso.gbest_value)

            # Write log file
            with open('log.txt', 'w') as f3:
                f3.write('{}. Random seed: {}, Best positon: {}, Best fitness: {}\n'.format(i, rand, pso.gbest_position, pso.gbest_value))

        # Write mean to file
        mean = np.mean(res)    
        std = np.std(res)
        with open('result_after_10_times.txt', 'w') as f4:
            f4.write('{}({})'.format(mean, std))

# Close all files
f.close()
f2.close()
f3.close()
f4.close()