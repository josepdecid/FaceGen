import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool
import os

import numpy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.train_constants import DEVICE, IMG_SIZE
from models.evolutionary import GARI
from models.evolutionary.face_classifier import FaceClassifier

"""
Reproduce a single image using Genetic Algorithm (GA) by evolving 
single pixel values.

This project works with both color and gray images without any modifications. 
Just give the image path.
Using three parameters, we can customize it to statisfy our need. 
The parameters are:
    1) Population size. I.e. number of individuals pepr population.
    2) Mating pool size. I.e. Number of selected parents in the mating pool.
    3) Mutation percentage. I.e. number of genes to change their values.

Value encoding used for representing the input.
Crossover is applied by exchanging half of genes from two parents.
Mutation is applied by randomly changing the values of randomly selected 
predefined percent of genes from the parents chromosome.
"""


class GeneticAlgorithm:
    def __init__(self, model: FaceClassifier, par=False, log_tag=None):
        self.model = model
        self.par = par
        self.log_tag = log_tag

        self.target_shape = (3, IMG_SIZE, IMG_SIZE)
        # Population size
        self.sol_per_pop = 100
        # Mating pool size
        self.num_parents_mating = 25
        # Mutation percentage
        self.mutation_percent = 0.10
        # Iterations
        self.generations = 1000000
        self.generations_until_merge = 10

        self.writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])

        # There might be inconsistency between the number of selected mating parents and
        # number of selected individuals within the population.
        # In some cases, the number of mating parents are not sufficient to
        # reproduce a new generation. If that occurred, the program will stop.
        possible_perm = len(list(itertools.permutations(numpy.arange(0, self.num_parents_mating), r=2)))
        required_perm = self.sol_per_pop - possible_perm
        if required_perm > possible_perm:
            raise AttributeError('Inconsistency in the selected population size or number of parents.')

    def run(self):
        self.model = self.model.to(DEVICE)

        if self.par:
            # Creating an initial population randomly.
            cores = multiprocessing.cpu_count()
            new_populations = [GARI.initial_population(n_individuals=self.sol_per_pop) for _ in range(cores)]

            for i in range(1, self.generations // self.generations_until_merge + 1):
                pool = ThreadPool(cores)
                lock = multiprocessing.Lock()

                next_populations = []
                next_fitness = [[] for _ in range(cores)]
                for c, new_population in enumerate(new_populations):
                    result = pool.apply_async(self._run_n_generations,
                                              args=(c, self.generations_until_merge, new_population, lock))
                    next_populations.append(result)

                pool.close()
                pool.join()

                for idx, result in enumerate(next_populations):
                    result.wait()
                    next_populations[idx], next_fitness[idx] = result.get()

                for it in range(self.generations_until_merge):
                    self.writer.add_scalars('Fitness value',
                                            {f'C{c}': next_fitness[c][it].max()
                                             for c in range(len(next_fitness))},
                                            (i - 1) * self.generations_until_merge + it)

                    next_merged_population = numpy.concatenate(next_populations, axis=0)

                    fitness_value = GARI.cal_pop_fitness(next_merged_population, model=self.model)
                    new_population = GARI.select_mating_pool(population=next_merged_population,
                                                             qualities=fitness_value,
                                                             num_parents=self.num_parents_mating)
                    GARI.save_images(i, new_population, self.model,
                                     save_point=i, save_dir=os.environ['CKPT_DIR'], log_tag=self.log_tag)
                    new_populations = [numpy.copy(new_population) for _ in range(cores)]
        else:
            new_population = GARI.initial_population(n_individuals=self.sol_per_pop)

            new_population, _ = self._run_n_generations(0, self.generations, new_population)
            # Display the final generation
            # GARI.show_indivs(new_population, target_shape)
            GARI.save_images(self.generations, new_population, self.model,
                             save_point=self.generations, save_dir=os.environ['CKPT_DIR'], log_tag=self.log_tag)

    def _run_n_generations(self, c: int, generations: int, new_population: numpy.ndarray, lock=None):
        if lock is not None:
            with lock:
                pb = tqdm(total=generations, desc=f'Thread {c}', ncols=100, position=c)
        else:
            pb = tqdm(total=generations, desc=f'Generations', ncols=100)

        fitness_values = []

        for generation in range(generations):
            fitness_value = GARI.cal_pop_fitness(new_population, model=self.model)
            fitness_values.append(fitness_value)
            # print(f'Fitness : {numpy.max(fitness_value)}, Iteration : {generation}')

            # Selecting the best parents in the population for mating.
            parents = GARI.select_mating_pool(population=new_population,
                                              qualities=fitness_value,
                                              num_parents=self.num_parents_mating)

            # Generating next generation using crossover.
            new_population = GARI.crossover(parents,
                                            n_individuals=self.sol_per_pop)

            new_population = GARI.mutation(population=new_population,
                                           num_parents_mating=self.num_parents_mating,
                                           mut_percent=self.mutation_percent)

            """
            Save best individual in the generation as an image for later visualization.
            """
            GARI.save_images(generation, new_population, self.model,
                             save_point=500, save_dir=os.environ['CKPT_DIR'], log_tag=self.log_tag)

            if lock is not None:
                with lock:
                    pb.update()
            else:
                self.writer.add_scalar('Fitness value', fitness_value.max(), global_step=generation)
                pb.update()

        if lock is not None:
            with lock:
                pb.close()
        else:
            pb.close()

        return new_population, fitness_values
