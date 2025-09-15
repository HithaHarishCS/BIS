import numpy as np
import math

def total_completion_time(order, processing_times):
    """Calculate total completion time given job order."""
    completion_time = 0
    current_time = 0
    for job in order:
        current_time += processing_times[job]
        completion_time += current_time
    return completion_time

def levy_flight(Lambda, dim):
    """Generate step sizes for Lévy flight"""
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def simple_bounds(s, lb, ub):
    """Keep values within bounds"""
    return np.clip(s, lb, ub)

def decode_solution(vector):
    """Decode continuous vector into a job permutation by sorting indices"""
    return np.argsort(vector)

def cuckoo_search_job_scheduling(processing_times, n_nests=25, pa=0.25, n_iterations=500):
    n_jobs = len(processing_times)
    lb, ub = 0, 1  # Bounds for continuous encoding
    
    # Initialize nests: random continuous vectors representing job orders
    nests = np.random.uniform(lb, ub, (n_nests, n_jobs))
    
    # Evaluate initial fitnesses
    fitness = np.array([total_completion_time(decode_solution(n), processing_times) for n in nests])
    
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    Lambda = 1.5
    
    for iteration in range(n_iterations):
        for i in range(n_nests):
            step = levy_flight(Lambda, n_jobs)
            step_size = 0.01 * step * (nests[i] - best_nest)
            new_solution = nests[i] + step_size
            new_solution = simple_bounds(new_solution, lb, ub)
            
            new_order = decode_solution(new_solution)
            new_fitness = total_completion_time(new_order, processing_times)
            
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness
                
                if new_fitness < best_fitness:
                    best_nest = new_solution.copy()
                    best_fitness = new_fitness
        
        # Abandon a fraction of worse nests and replace them with random ones
        n_abandon = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[-n_abandon:]
        for idx in worst_indices:
            nests[idx] = np.random.uniform(lb, ub, n_jobs)
            fitness[idx] = total_completion_time(decode_solution(nests[idx]), processing_times)
        
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}: Best total completion time = {best_fitness}")
    
    best_order = decode_solution(best_nest)
    return best_order, best_fitness

# Example usage
processing_times = [3, 1, 7, 5, 2]  # Processing times of 5 jobs

best_order, best_time = cuckoo_search_job_scheduling(processing_times, n_nests=30, pa=0.25, n_iterations=500)
print("Best job order (0-indexed):", best_order)
print("Best total completion time:", best_time)
