import numpy as np
import pandas as pd
import random
import time
import csv

# Загрузка матрицы расстояний
matrix = pd.read_csv('distance_matrix.csv', index_col=0).values
N = matrix.shape[0]

def run_ant_colony(alpha, beta, rho, q, ants, iters):
    """Запускает алгоритм муравьиной колонии и возвращает метрики."""
    pheromone = np.ones((N, N))
    safe_matrix = matrix + np.eye(N)
    eta = 1 / safe_matrix
    eta[matrix == 0] = 1e-10

    best_length = float('inf')
    best_route = None
    start_time = time.perf_counter()
    max_iter = 0
    min_iter = float('inf')

    for _ in range(iters):
        t0 = time.perf_counter()
        all_lengths = []
        all_routes = []

        for _ in range(ants):
            unvisited = set(range(N))
            route = [random.choice(list(unvisited))]
            unvisited.remove(route[0])

            while unvisited:
                current = route[-1]
                probs = []
                for city in unvisited:
                    probs.append(
                        (pheromone[current, city] ** alpha) *
                        (eta[current, city] ** beta)
                    )
                probs = np.array(probs)
                if not np.isfinite(probs.sum()) or probs.sum() == 0:
                    choice = random.choice(list(unvisited))
                else:
                    probs /= probs.sum()
                    choice = random.choices(list(unvisited), weights=probs)[0]
                route.append(choice)
                unvisited.remove(choice)

            length = sum(matrix[route[i], route[(i+1)%N]] for i in range(N))
            all_routes.append(route)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_route = route.copy()

        # обновление феромона
        pheromone *= (1 - rho)
        for route, length in zip(all_routes, all_lengths):
            deposit = q / length
            for i in range(N):
                a, b = route[i], route[(i+1)%N]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        dt = time.perf_counter() - t0
        max_iter = max(max_iter, dt)
        min_iter = min(min_iter, dt)

    total_time = time.perf_counter() - start_time
    return {
        'best_route': [i+1 for i in best_route],
        'best_length': best_length,
        'total_time': total_time,
        'max_iter_time': max_iter,
        'min_iter_time': min_iter
    }

ITERS_PER_PARAM = 50

# --- Задаём пакеты параметров ---
param_batches = [
    {'alpha': 0.5, 'beta': 2.0,  'rho': 0.2, 'q': 50,  'num_ants': 30, 'num_iter': 30},
    {'alpha': 1.0, 'beta': 5.0,  'rho': 0.5, 'q': 100, 'num_ants': 50, 'num_iter': 80},
    {'alpha': 1.0, 'beta': 5.0,  'rho': 0.5, 'q': 100, 'num_ants': 50, 'num_iter': 100},
    {'alpha': 1.0, 'beta': 5.0,  'rho': 0.5, 'q': 100, 'num_ants': 50, 'num_iter': 200},
    # можно добавить ещё…
]

detailed_file = 'detailed_results.csv'
summary_file  = 'batch_summary.csv'

detail_fields = [
    'alpha','beta','rho','q','num_ants','num_iter',
    'seed','best_length','total_time','max_iter_time','min_iter_time','best_route'
]
summary_fields = [
    'alpha','beta','rho','q','num_ants','num_iter',
    'avg_best_length','avg_total_time'
]

with open(detailed_file, 'w', newline='') as df, open(summary_file, 'w', newline='') as sf:
    detail_writer = csv.DictWriter(df, fieldnames=detail_fields)
    summary_writer = csv.DictWriter(sf,  fieldnames=summary_fields)
    detail_writer.writeheader()
    summary_writer.writeheader()

    iters = 0

    for params in param_batches:
        total_times = []
        best_lengths = []

        for _ in range(ITERS_PER_PARAM):
            # генерируем случайный seed и фиксируем его
            seed = random.randrange(0, 2**32)
            random.seed(seed)
            np.random.seed(seed)

            res = run_ant_colony(
                alpha     = params['alpha'],
                beta      = params['beta'],
                rho       = params['rho'],
                q         = params['q'],
                ants      = params['num_ants'],
                iters     = params['num_iter']
            )

            row = {
                **params,
                'seed': seed,
                'best_length': res['best_length'],
                'total_time': res['total_time'],
                'max_iter_time': res['max_iter_time'],
                'min_iter_time': res['min_iter_time'],
                'best_route': res['best_route']
            }
            detail_writer.writerow(row)
            df.flush()

            total_times.append(res['total_time'])
            best_lengths.append(res['best_length'])

        iters += ITERS_PER_PARAM    

        summary = {
            **params,
            'avg_best_length': sum(best_lengths) / len(best_lengths),
            'avg_total_time':   sum(total_times)  / len(total_times)
        }
        summary_writer.writerow(summary)
        sf.flush()

        print(f"Успешно сохранены: {iters}")

print("Готово! Детали — в", detailed_file, "; сводка — в", summary_file)
