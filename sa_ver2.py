import numpy as np
import pandas as pd
import random
import time
from itertools import product

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

# Параметры для перебора
NUM_RUNS = 50  # Количество запусков
PARAMS = {
    'INITIAL_TEMPERATURE': [1000, 30000, 100000],
    'ALPHA': [0.99, 0.99985, 0.9999],
    'SWAP_PROBABILITY': [0.2, 0.4, 0.6],
    'REVERSE_PROBABILITY': [0.4, 0.6, 0.8]
}
FINAL_TEMPERATURE = 1e-6
MAX_ITER = 15_000_000
LOG_INTERVAL = 10000

# Функции алгоритма
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

# Функция для одного запуска алгоритма
def run_simulated_annealing(initial_temperature, alpha, swap_probability, reverse_probability):
    current_route = list(range(N))
    random.shuffle(current_route)
    current_length = route_length(current_route)

    best_route = current_route.copy()
    best_length = current_length

    temperature = initial_temperature
    iteration = 0
    max_iteration_duration = 0
    min_iteration_duration = float("inf")
    algo_start_time = time.perf_counter()

    while temperature > FINAL_TEMPERATURE and iteration < MAX_ITER:
        iteration_start_time = time.perf_counter()

        r = random.random()
        if r < swap_probability:
            candidate_route = random_swap(current_route)
        elif r < reverse_probability:
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

        temperature *= alpha
        iteration += 1

        iteration_end_time = time.perf_counter()
        iteration_duration = iteration_end_time - iteration_start_time
        max_iteration_duration = max(max_iteration_duration, iteration_duration)
        min_iteration_duration = min(min_iteration_duration, iteration_duration)

        if iteration % LOG_INTERVAL == 0:
            print(f"Итерация {iteration}: текущая длина = {current_length}, лучшая = {best_length}, температура = {temperature:.4f}")

    algo_end_time = time.perf_counter()

    return {
        'best_route': [i + 1 for i in best_route],
        'best_length': best_length,
        'total_time': algo_end_time - algo_start_time,
        'max_iter_time': max_iteration_duration,
        'min_iter_time': min_iteration_duration
    }

# Перебор комбинаций параметров
param_combinations = list(product(PARAMS['INITIAL_TEMPERATURE'], PARAMS['ALPHA'], PARAMS['SWAP_PROBABILITY'], PARAMS['REVERSE_PROBABILITY']))

# Запуск и сохранение результатов
with open('simulated_annealing_results.txt', 'w') as f:
    for run in range(NUM_RUNS):

        random_num = random.randint(0, 81)
        initial_temperature, alpha, swap_probability, reverse_probability = param_combinations[random_num]

        print(f"\nЗапуск {run + 1} с параметрами: INITIAL_TEMPERATURE={initial_temperature}, ALPHA={alpha}, SWAP_PROBABILITY={swap_probability}, REVERSE_PROBABILITY={reverse_probability}")
        result = run_simulated_annealing(initial_temperature, alpha, swap_probability, reverse_probability)

        # Запись в файл
        f.write(f"Запуск {run + 1}\n")
        f.write(f"Параметры: INITIAL_TEMPERATURE={initial_temperature}, ALPHA={alpha}, SWAP_PROBABILITY={swap_probability}, REVERSE_PROBABILITY={reverse_probability}\n")
        f.write(f"Лучший маршрут: {result['best_route']}\n")
        f.write(f"Длина маршрута: {result['best_length']}\n")
        f.write(f"Общая длительность: {result['total_time']:.2f} секунд\n")
        f.write(f"Длительность итерации: max={result['max_iter_time']:.6f}, min={result['min_iter_time']:.6f} секунд\n")
        f.write("-" * 50 + "\n")

print("Результаты сохранены в simulated_annealing_results.txt")
