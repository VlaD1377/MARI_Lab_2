import numpy as np
import time
from pso import PSO
from functions import dixon_price
from optimal_solver import DixonPriceExact

dim = 10
runs = 30
topologies = ["gbest","ring","grid"]

results = []
positions = []
pso_instances = []
times = []
seeds = []

for run in range(runs):
    seed = np.random.randint(0, 10000000)
    np.random.seed(seed)

    pso = PSO(
        dixon_price,
        dim,
        min=-10,
        max=10,
        particles=100,
        w=0.6,
        c1=1.8,
        c2=1.2,
        iterations=200,
        topology=topologies[0]
    )

    start = time.time()
    x, f = pso.solve()
    end = time.time()

    results.append(f)
    positions.append(x)
    pso_instances.append(pso)
    times.append(end - start)
    seeds.append(seed)

    print(f"Run {run + 1}: seed = {seed}, f = {f}, time = {end - start:.4f} s")

results = np.array(results)
times = np.array(times)

best_index = np.argmin(results)
best_f = results[best_index]
best_x = positions[best_index]
best_pso = pso_instances[best_index]
best_time = times[best_index]
best_seed = seeds[best_index]

print("\n=== Statistical results ===")
print("Best seed:", best_seed)
print("Best f:", best_f)
print("Best x:", best_x)
print("Mean f:", results.mean())
print("Std f:", results.std())
print(f"Best run time: {best_time:.4f} s")
print(f"Mean time: {times.mean():.4f} s")

print("\n=== Optimal Solver ===")
exact_solver = DixonPriceExact(dim)
x_opt, f_opt = exact_solver.solve()
print("Optimal f:", f_opt)
print("Optimal x:", x_opt)

# print("\n=== Best run iterations ===")
# best_pso.print_iterations()
best_pso.plot_convergence()

best_pso.animate_optimization()