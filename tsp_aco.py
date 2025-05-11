import numpy as np
import pandas as pd
import random

import time

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

# Параметры муравьиного алгоритма
NUM_ANTS = 50
NUM_ITER = 200
ALPHA = 1.0  # влияние феромона
BETA = 5.0   # влияние расстояния
RHO = 0.5    # коэффициент испарения феромона
Q = 100      # количество феромона, выделяемого муравьем

# Параметр частоты промежуточного вывода

LOG_INTERVAL = 50

max_iteration_duration = 0
min_iteration_duration = float("inf")

algo_start_time = time.perf_counter()

# Инициализация феромонов
pheromone = np.ones((N, N))

# Предрасчёт обратных расстояний
eta = 1 / (matrix + np.eye(N))  # избегаем деления на 0 на диагонали
eta[matrix == 0] = 0

best_length = float('inf')
best_route = None

for iteration in range(NUM_ITER):
    iteration_start_time = time.perf_counter()

    all_routes = []
    all_lengths = []
    for ant in range(NUM_ANTS):
        unvisited = set(range(N))
        route = [random.choice(list(unvisited))]
        unvisited.remove(route[0])
        while unvisited:
            current = route[-1]
            probabilities = []
            for city in unvisited:
                tau = pheromone[current, city] ** ALPHA
                et = eta[current, city] ** BETA
                probabilities.append(tau * et)
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            next_city = random.choices(list(unvisited), weights=probabilities)[0]
            route.append(next_city)
            unvisited.remove(next_city)
        all_routes.append(route)
        length = sum(matrix[route[i], route[(i+1)%N]] for i in range(N))
        all_lengths.append(length)
        if length < best_length:
            best_length = length
            best_route = route.copy()
    # Испарение феромона
    pheromone *= (1 - RHO)
    # Добавление феромона
    for route, length in zip(all_routes, all_lengths):
        for i in range(N):
            a, b = route[i], route[(i+1)%N]
            pheromone[a, b] += Q / length
            pheromone[b, a] += Q / length

    iteration_end_time = time.perf_counter()
    iteration_duration = iteration_end_time - iteration_start_time
    max_iteration_duration = max(max_iteration_duration, iteration_duration)
    min_iteration_duration = min(min_iteration_duration, iteration_duration)

    if iteration % LOG_INTERVAL == 0:
        print(f"Итерация {iteration}: лучшая длина = {best_length}")

algo_end_time = time.perf_counter()        

# Вывод результата
best_route = [i + 1 for i in best_route]
print("Лучший маршрут:", best_route)
print("Длина маршрута:", best_length) 
print(f"Общая длительность выполнения алгоритма: {(algo_end_time - algo_start_time):.2f} секунд")
print(f"Длителность выполнения итерации: max: {max_iteration_duration:.6f} секунд, min: {min_iteration_duration:.6f} секунд")
