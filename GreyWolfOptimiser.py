import numpy as np
import random

# Distance matrix
def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return dist

# Fitness = total path length
def fitness(tour, dist):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

# Grey Wolf Optimizer for TSP
def gwo_tsp(cities, pop_size=20, max_iter=200):
    n = len(cities)
    dist = distance_matrix(cities)

    # Initialize wolves (random tours)
    wolves = [random.sample(range(n), n) for _ in range(pop_size)]
    scores = [fitness(w, dist) for w in wolves]

    # Identify leaders
    idx = np.argsort(scores)
    alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
    alpha_score, beta_score, delta_score = scores[idx[0]], scores[idx[1]], scores[idx[2]]

    # Remaining are omegas
    omegas = [wolves[i] for i in idx[3:]]

    # Main loop
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # linearly decreases from 2 → 0
        new_wolves = []

        # Update omegas using alpha, beta, delta influence
        for wolf in omegas:
            new_tour = wolf[:]
            for i in range(n):
                r1, r2, r3 = random.random(), random.random(), random.random()
                A1, C1 = 2*a*r1 - a, 2*r1
                A2, C2 = 2*a*r2 - a, 2*r2
                A3, C3 = 2*a*r3 - a, 2*r3

                # Influences
                X1, X2, X3 = alpha[i % n], beta[i % n], delta[i % n]

                # Pick a leader influence
                choice = random.choice([X1, X2, X3])
                if choice in new_tour:
                    j = new_tour.index(choice)
                    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

            new_wolves.append(new_tour)

        # Combine leaders + new omegas
        wolves = [alpha, beta, delta] + new_wolves
        scores = [fitness(w, dist) for w in wolves]

        # Update leaders
        idx = np.argsort(scores)
        alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
        alpha_score, beta_score, delta_score = scores[idx[0]], scores[idx[1]], scores[idx[2]]
        omegas = [wolves[i] for i in idx[3:]]

    return alpha, alpha_score

# Example
if __name__ == "__main__":
    cities = [(0,0), (1,5), (5,2), (6,6), (8,3)]
    best_tour, best_distance = gwo_tsp(cities, pop_size=15, max_iter=100)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)
