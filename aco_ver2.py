import numpy as np
import pandas as pd
import random
import time
from itertools import product

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

# Параметры для перебора
NUM_RUNS = 2160  # Количество запусков
PARAMS = {
    'ALPHA': [0.5, 1.0, 2.0],
    'BETA': [2.0, 5.0, 8.0],
    'RHO': [0.2, 0.5, 0.8],
    'Q': [50, 100, 150, 200],
    'NUM_ANTS': [30, 50, 80, 100],
    'NUM_ITER': [30, 80, 100, 200, 300],
}
param_combinations = list(product(
    PARAMS['ALPHA'],
    PARAMS['BETA'],
    PARAMS['RHO'],
    PARAMS['Q'],
    PARAMS['NUM_ANTS'],
    PARAMS['NUM_ITER']
))

# LOG_INTERVAL = 10

# Функция для одного запуска алгоритма
def run_ant_colony(alpha, beta, rho, q, ants, iters):
    pheromone = np.ones((N, N))

    # Обработка матрицы расстояний для eta
    safe_matrix = matrix + np.eye(N)  # защита от деления на 0 на диагонали
    eta = 1 / safe_matrix
    eta[matrix == 0] = 1e-10  # чтобы не было нулей в eta

    best_length = float('inf')
    best_route = None

    algo_start_time = time.perf_counter()
    max_iteration_duration = 0
    min_iteration_duration = float("inf")

    for iteration in range(iters):
        iteration_start_time = time.perf_counter()
        all_routes = []
        all_lengths = []

        for ant in range(ants):
            unvisited = set(range(N))
            route = [random.choice(list(unvisited))]
            unvisited.remove(route[0])

            while unvisited:
                current = route[-1]
                probabilities = []

                for city in unvisited:
                    tau = pheromone[current, city] ** alpha
                    et = eta[current, city] ** beta
                    probabilities.append(tau * et)

                probabilities = np.array(probabilities)
                total = probabilities.sum()

                if total == 0 or not np.isfinite(total):
                    next_city = random.choice(list(unvisited))
                else:
                    probabilities /= total
                    next_city = random.choices(list(unvisited), weights=probabilities)[0]

                route.append(next_city)
                unvisited.remove(next_city)

            all_routes.append(route)
            length = sum(matrix[route[i], route[(i + 1) % N]] for i in range(N))
            all_lengths.append(length)
            if length < best_length:
                best_length = length
                best_route = route.copy()

        # Обновление феромонов
        pheromone *= (1 - rho)
        for route, length in zip(all_routes, all_lengths):
            for i in range(N):
                a, b = route[i], route[(i + 1) % N]
                pheromone[a, b] += q / length
                pheromone[b, a] += q / length

        iteration_end_time = time.perf_counter()
        iteration_duration = iteration_end_time - iteration_start_time
        max_iteration_duration = max(max_iteration_duration, iteration_duration)
        min_iteration_duration = min(min_iteration_duration, iteration_duration)

        # if iteration % LOG_INTERVAL == 0:
        #     print(f"Итерация {iteration}: лучшая длина = {best_length}")

    algo_end_time = time.perf_counter()
    return {
        'best_route': [i + 1 for i in best_route],
        'best_length': best_length,
        'total_time': algo_end_time - algo_start_time,
        'max_iter_time': max_iteration_duration,
        'min_iter_time': min_iteration_duration
    }

# Перебор комбинаций параметров
# param_combinations = list(product(PARAMS['ALPHA'], PARAMS['BETA'], PARAMS['RHO'], PARAMS['Q'], PARAMS['NUM_ANTS'], PARAMS['NUM_ITER']))

# Запуск и сохранение результатов
with open('ant_colony_results.txt', 'w') as f:
    for run, (alpha, beta, rho, q, num_ants, num_iter) in enumerate(param_combinations, start=1):
        print(f"\nЗапуск {run}")
        result = run_ant_colony(alpha, beta, rho, q, num_ants, num_iter)

        f.write(f"Запуск {run}\n")
        f.write(f"Параметры: ALPHA={alpha}, BETA={beta}, RHO={rho}, Q={q}, ANTS={num_ants}, ITERS={num_iter}\n")
        f.write(f"Лучший маршрут: {result['best_route']}\n")
        f.write(f"Длина маршрута: {result['best_length']}\n")
        f.write(f"Общая длительность: {result['total_time']:.2f} секунд\n")
        f.write(f"Длительность итерации: max={result['max_iter_time']:.6f}, min={result['min_iter_time']:.6f} секунд\n")
        f.write("-" * 50 + "\n")

print("Результаты сохранены в ant_colony_results.txt")
