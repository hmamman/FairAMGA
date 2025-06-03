import numpy as np


class AMGA:
    def __str__(self):
        return "AMGA"

    def __init__(self, pop_size, dna_size, bound, discrimination_check, cross_rate=0.8,
                 mutation_rate=0.1, mutation_update_interval=10, sigma=0.1, lambda_=10):
        self.gen = 0
        self.N = pop_size
        self.n = dna_size
        self.bound = np.array(bound)
        self.fitness_func = discrimination_check
        self.cr = cross_rate
        self.mr = mutation_rate
        self.mui = mutation_update_interval
        self.sigma = sigma
        self.lambda_ = lambda_

        self.population = None
        self.best_fitness_history = []
        self.population_diversity = []
        self.mutation_rate_history = []
        self.elite_size = int(self.N * 0.1)  # 10% elitism
        self.tournament_size = max(5, int(self.N * 0.05))

        self.generate_initial_population()

    def generate_initial_population(self):
        self.population = np.random.uniform(
            low=self.bound[:, 0],
            high=self.bound[:, 1],
            size=(self.N, self.n)
        )

    def get_fitness(self):
        # Add a small epsilon value to avoid division by zero
        epsilon = 1e-10
        fitness = np.array([self.fitness_func(self.population[i]) for i in range(len(self.population))]) + epsilon
        return fitness

    def select(self):
        fitness = self.get_fitness()

        # Elitism selection Select the best individuals from the population (based on fitness) and keep them
        # unchanged in the next generation. The number of selected individuals depends on the elitism size (10% of
        # the population). The selected individuals are then combined with the rest of the population to form the
        # next generation.
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = self.population[elite_indices]

        # Tournament selection for the rest
        selected = []
        for _ in range(self.N - self.elite_size):
            tournament = np.random.choice(np.arange(self.N), self.tournament_size, replace=False)
            winner = tournament[np.argmax(fitness[tournament])]
            selected.append(self.population[winner])

        self.population = np.vstack((elite, selected))

    def crossover(self):
        for i in range(0, self.N, 2):
            if np.random.rand() < self.cr:
                crossover_point = np.random.randint(1, self.n)
                self.population[i, crossover_point:], self.population[i + 1, crossover_point:] = \
                    self.population[i + 1, crossover_point:].copy(), self.population[i, crossover_point:].copy()

    def mutate(self):
        # Adaptive mutation rate
        if self.gen % self.lambda_ == 0:
            self.update_mutation()

        for i in range(self.elite_size, self.N):  # Don't mutate elite individuals
            for j in range(self.n):
                if np.random.rand() < self.mr:
                    delta = (self.bound[j, 1] - self.bound[j, 0]) * self.sigma
                    self.population[i, j] += np.random.uniform(-delta, delta)
                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])

    def update_mutation(self):
        diversity = self.calculate_diversity()
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        # Clamp mutation rate to reasonable bounds
        self.mr = np.clip(self.sigma / (diversity + epsilon), 0.001, 0.5)

    def calculate_diversity(self):
        return np.mean(np.std(self.population, axis=0))

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()

        self.gen += 1
