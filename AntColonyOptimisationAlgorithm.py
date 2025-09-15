import numpy as np
import random
import math

# -----------------------------
# STEP 1: Define cities (coordinates)
# -----------------------------
cities = {
    0: (0, 0),
    1: (2, 6),
    2: (5, 3),
    3: (6, 7),
    4: (8, 2)
}
num_cities = len(cities)

# Print input
print("📌 INPUT: City Coordinates")
for city, coord in cities.items():
    print(f"  City {city}: {coord}")
print("-" * 40)

# -----------------------------
# STEP 2: Create distance matrix
# -----------------------------
def euclidean_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            dist_matrix[i][j] = euclidean_distance(cities[i], cities[j])

# -----------------------------
# STEP 3: ACO Parameters
# -----------------------------
num_ants = 10
num_iterations = 100
alpha = 1
beta = 5
rho = 0.5
Q = 100
pheromone = np.ones((num_cities, num_cities))

# -----------------------------
# STEP 4: Construct tour
# -----------------------------
def construct_solution():
    path = []
    visited = set()
    current = random.randint(0, num_cities - 1)
    path.append(current)
    visited.add(current)

    while len(path) < num_cities:
        probabilities = []
        for j in range(num_cities):
            if j not in visited:
                tau = pheromone[current][j] ** alpha
                eta = (1 / dist_matrix[current][j]) ** beta
                probabilities.append((j, tau * eta))

        total = sum(prob for _, prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]

        next_city = random.choices(
            [city for city, _ in probabilities],
            [prob for _, prob in probabilities]
        )[0]
        path.append(next_city)
        visited.add(next_city)
        current = next_city

    return path

# -----------------------------
# STEP 5: Path distance + edge weights
# -----------------------------
def path_length(path):
    total = 0
    for i in range(len(path) - 1):
        total += dist_matrix[path[i]][path[i + 1]]
    total += dist_matrix[path[-1]][path[0]]  # Return to start
    return total

def get_edge_weights(path):
    edges = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edges.append((a, b, dist_matrix[a][b]))
    # Return to start
    a, b = path[-1], path[0]
    edges.append((a, b, dist_matrix[a][b]))
    return edges

# -----------------------------
# STEP 6: ACO Main Loop
# -----------------------------
best_path = None
best_length = float('inf')

for _ in range(num_iterations):
    all_paths = []
    all_lengths = []

    for _ in range(num_ants):
        path = construct_solution()
        length = path_length(path)
        all_paths.append(path)
        all_lengths.append(length)

        if length < best_length:
            best_length = length
            best_path = path

    pheromone *= (1 - rho)
    for path, length in zip(all_paths, all_lengths):
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % num_cities]
            pheromone[from_city][to_city] += Q / length
            pheromone[to_city][from_city] += Q / length

# -----------------------------
# STEP 7: Final Output
# -----------------------------
print("\n🎯 OUTPUT: Best Tour Found")

# Show path
print("  ➤ Tour Path (city numbers):")
print("    " + " -> ".join(str(city) for city in best_path) + f" -> {best_path[0]}")

# Show edges and weights
print("\n  ➤ Edges and Weights:")
edge_list = get_edge_weights(best_path)
for a, b, w in edge_list:
    print(f"    {a} -> {b} : {w:.2f}")

# Show total distance
print(f"\n📏 Total Distance: {best_length:.2f} units")
