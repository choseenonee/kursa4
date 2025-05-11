import numpy as np
import pandas as pd
import random

import time

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

# Параметры генетического алгоритма
POP_SIZE = 200
N_GENERATIONS = 2000
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5

# Параметр частоты промежуточного вывода

LOG_INTERVAL = 100

def create_route():
    route = list(range(N))
    random.shuffle(route)
    return route


def route_length(route):
    return sum(matrix[route[i], route[(i + 1) % N]] for i in range(N))


def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(N), 2))
    child = [None] * N
    child[start:end] = parent1[start:end]
    ptr = end
    for city in parent2:
        if city not in child:
            if ptr >= N:
                ptr = 0
            child[ptr] = city
            ptr += 1
    return child


def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        route[i], route[j] = route[j], route[i]
    return route


def tournament_selection(population, fitnesses):
    selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]


max_iteration_duration = 0
min_iteration_duration = float("inf")

algo_start_time = time.perf_counter()

# Инициализация популяции
population = [create_route() for _ in range(POP_SIZE)]

for generation in range(N_GENERATIONS):
    iteration_start_time = time.perf_counter()

    fitnesses = [route_length(route) for route in population]
    new_population = []
    for _ in range(POP_SIZE):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    population = new_population

    iteration_end_time = time.perf_counter()
    iteration_duration = iteration_end_time - iteration_start_time
    max_iteration_duration = max(max_iteration_duration, iteration_duration)
    min_iteration_duration = min(min_iteration_duration, iteration_duration)

    if generation % LOG_INTERVAL == 0:
        print(f"Поколение {generation}: лучший маршрут = {min(fitnesses)}")

# Лучший найденный маршрут
fitnesses = [route_length(route) for route in population]
best_idx = np.argmin(fitnesses)

algo_end_time = time.perf_counter()

best_route = population[best_idx]
best_route = [i + 1 for i in best_route]
print("Лучший маршрут:", best_route)
print("Длина маршрута:", fitnesses[best_idx]) 
print(f"Общая длительность выполнения алгоритма: {(algo_end_time - algo_start_time):.2f} секунд")
print(f"Длителность выполнения итерации: max: {max_iteration_duration:.6f} секунд, min: {min_iteration_duration:.6f} секунд")
