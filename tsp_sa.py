import numpy as np
import pandas as pd
import random

import time

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

# Параметры имитации отжига
INITIAL_TEMPERATURE = 30000
FINAL_TEMPERATURE = 1e-6
ALPHA = 0.99985
MAX_ITER = 15_000_000

# Параметры вероятностей мутаций (см. использование)

SWAP_PROBABILITY = 0.4
REVERSE_PROBABILITY = 0.8

# Параметр частоты промежуточного вывода

LOG_INTERVAL = 10000

def route_length(route):
    return sum(matrix[route[i], route[(i + 1) % N]] for i in range(N))

def random_swap(route):
    i, j = random.sample(range(N), 2)
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def random_reverse(route):
    i, j = sorted(random.sample(range(N), 2))
    new_route = route.copy()
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

def random_insert(route):
    i, j = random.sample(range(N), 2)
    new_route = route.copy()
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route

# Начальное решение
current_route = list(range(N))
random.shuffle(current_route)
current_length = route_length(current_route)

best_route = current_route.copy()
best_length = current_length

temperature = INITIAL_TEMPERATURE
iteration = 0

max_iteration_duration = 0
min_iteration_duration = float("inf")

algo_start_time = time.perf_counter()

while temperature > FINAL_TEMPERATURE and iteration < MAX_ITER:
    iteration_start_time = time.perf_counter()
    # С вероятностями используем разные мутации
    r = random.random()
    if r < SWAP_PROBABILITY:
        candidate_route = random_swap(current_route)
    elif r < REVERSE_PROBABILITY:
        candidate_route = random_reverse(current_route)
    else:
        candidate_route = random_insert(current_route)

    candidate_length = route_length(candidate_route)
    delta = candidate_length - current_length

    if delta < 0 or random.random() < np.exp(-delta / temperature):
        current_route = candidate_route
        current_length = candidate_length
        if current_length < best_length:
            best_route = current_route.copy()
            best_length = current_length

    if iteration % LOG_INTERVAL == 0:
        print(f"Итерация {iteration}: текущая длина = {current_length}, лучшая = {best_length}, температура = {temperature:.4f}")

    temperature *= ALPHA
    iteration += 1

    iteration_end_time = time.perf_counter()

    iteration_duration = iteration_end_time - iteration_start_time

    max_iteration_duration = max(max_iteration_duration, iteration_duration)
    min_iteration_duration = min(min_iteration_duration, iteration_duration)

algo_end_time = time.perf_counter()

# Вывод результата
best_route = [i + 1 for i in best_route]
print("Лучший маршрут:", best_route)
print("Длина маршрута:", best_length)
print(f"Общая длительность выполнения алгоритма: {(algo_end_time - algo_start_time):.2f} секунд")
print(f"Длителность выполнения итерации: max: {max_iteration_duration:.6f} секунд, min: {min_iteration_duration:.6f} секунд")
