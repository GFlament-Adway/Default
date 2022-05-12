import numpy as np
class gen_algo:
    def __init__(self, pop_size, n_params, func, alpha, mutate_rate):
        self.pop_size = pop_size
        self.n_params = n_params
        self.func = func
        self.alpha = alpha
        self.mutate_rate = mutate_rate
        self.fitness_function = lambda x: x
        self.current_gen = 0
        self.init_pop()

    def evaluate_fitness(self):
        self.current_fitness = [self.fitness_function(self.func([p for p in x])) for x in self.current_pop]

    def init_pop(self):
        self.current_pop = [[np.random.randint(-1, 1)*np.random.random() for _ in range(self.n_params)] for _ in range(self.pop_size)]

    def generate_next_gen(self):
        ordered_fitness = sorted(self.current_fitness, reverse = True)
        best_params = ordered_fitness[:int(self.pop_size*self.alpha)]
        new_pop = [self.current_pop[self.current_fitness.index(param)] for param in best_params]

        for _ in range(self.pop_size - int(self.pop_size*self.alpha)):
            pop = new_pop[np.random.randint(0, len(new_pop) - 1)]
            mom = new_pop[np.random.randint(0, len(new_pop) - 1)]
            if np.random.random() < self.mutate_rate:
                pop = [p + np.random.random()*np.random.randint(-5, 5) for p in pop]
            son = [pop[k] if np.random.random() < 0.5 else mom[k] for k in range(len(pop))]
            new_pop += [son]

        assert len(new_pop) == self.pop_size
        self.current_pop = new_pop

def f(params):
    return 2*params[0]**2 - params[0] + 3*params[1]**3 + 2*params[1]**(4)

if __name__ == "__main__":
    gen_algo = gen_algo(1000, 2, f, 0.5, 0.05)
    for _ in range(10):
        print(gen_algo.current_pop)
        gen_algo.evaluate_fitness()
        print(np.min(gen_algo.current_fitness))
        gen_algo.generate_next_gen()
        gen_algo.evaluate_fitness()
