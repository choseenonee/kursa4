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
    'MUTATION_RATE': [0.05, 0.1, 0.3],
    'TOURNAMENT_SIZE': [3, 5, 10],
    'POP_SIZE': [100, 300, 400],
    'N_GENERATIONS': [500, 1000, 1500, 2000],
    'ELITE_SIZE': [1, 3, 5]
}

LOG_INTERVAL = 100

# Функции алгоритма
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

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        route[i], route[j] = route[j], route[i]
    return route

def tournament_selection(population, fitnesses, tournament_size):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

# Функция для одного запуска алгоритма
def run_genetic_algorithm(pop_size, mutation_rate, tournament_size, n_gen, elite_size):
    random.seed(42)
    np.random.seed(42)

    population = [create_route() for _ in range(pop_size)]
    max_iteration_duration = 0
    min_iteration_duration = float("inf")
    algo_start_time = time.perf_counter()

    for generation in range(n_gen):
        iteration_start_time = time.perf_counter()

        fitnesses = [route_length(route) for route in population]
        elite_indices = np.argsort(fitnesses)[:elite_size]
        elites = [population[i] for i in elite_indices]  # теперь "elites" это маршруты

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        iteration_end_time = time.perf_counter()
        iteration_duration = iteration_end_time - iteration_start_time
        max_iteration_duration = max(max_iteration_duration, iteration_duration)
        min_iteration_duration = min(min_iteration_duration, iteration_duration)

        if generation % LOG_INTERVAL == 0:
            print(f"Поколение {generation}: лучший маршрут = {min(fitnesses)}")

    fitnesses = [route_length(route) for route in population]
    best_idx = np.argmin(fitnesses)
    algo_end_time = time.perf_counter()

    return {
        'best_route': [i + 1 for i in population[best_idx]],
        'best_length': fitnesses[best_idx],
        'total_time': algo_end_time - algo_start_time,
        'max_iter_time': max_iteration_duration,
        'min_iter_time': min_iteration_duration
    }

# Перебор комбинаций параметров
param_combinations = list(product(PARAMS['MUTATION_RATE'], PARAMS['TOURNAMENT_SIZE'], PARAMS['POP_SIZE'], PARAMS['N_GENERATIONS'], PARAMS['ELITE_SIZE']))

# Запуск и сохранение результатов
with open('genetic_algorithm_results.txt', 'w') as f:
    for run in range(NUM_RUNS):

        rand_num = random.randint(0, 540)
        mutation_rate, tournament_size, pop_size, gens, elites = param_combinations[rand_num]

        print(f"\nЗапуск {run + 1} с параметрами: MUTATION_RATE={mutation_rate}, TOURNAMENT_SIZE={tournament_size}, POP_SIZE={pop_size}, N_GENERATIONS={gens}, ELITE_SIZE={elites}")
        result = run_genetic_algorithm(pop_size, mutation_rate, tournament_size, gens, elites)

        # Запись в файл
        f.write(f"Запуск {run + 1}\n")
        f.write(f"Параметры: MUTATION_RATE={mutation_rate}, TOURNAMENT_SIZE={tournament_size}, POP_SIZE={pop_size}, N_GENERATIONS={gens}, ELITE_SIZE={elites}\n")
        f.write(f"Лучший маршрут: {result['best_route']}\n")
        f.write(f"Длина маршрута: {result['best_length']}\n")
        f.write(f"Общая длительность: {result['total_time']:.2f} секунд\n")
        f.write(f"Длительность итерации: max={result['max_iter_time']:.6f}, min={result['min_iter_time']:.6f} секунд\n")
        f.write("-" * 50 + "\n")

print("Результаты сохранены в genetic_algorithm_results.txt")
