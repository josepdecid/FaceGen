import itertools
import multiprocessing
import os
from multiprocessing.pool import ThreadPool

import numpy

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
    def __init__(self, model: FaceClassifier, par=False):
        self.model = model
        self.par = par

        self.target_shape = (200, 200, 3)
        # Population size
        self.sol_per_pop = 75
        # Mating pool size
        self.num_parents_mating = 10
        # Mutation percentage
        self.mutation_percent = 0.05
        # Iterations
        self.iterations = 10000

        # There might be inconsistency between the number of selected mating parents and
        # number of selected individuals within the population.
        # In some cases, the number of mating parents are not sufficient to
        # reproduce a new generation. If that occurred, the program will stop.
        possible_perm = len(list(itertools.permutations(numpy.arange(0, self.num_parents_mating), r=2)))
        required_perm = self.sol_per_pop - possible_perm
        if required_perm > possible_perm:
            raise AttributeError('Inconsistency in the selected population size or number of parents.')

    def run(self):
        if self.par:
            cores = min(multiprocessing.cpu_count(), 5)
            pool = ThreadPool(cores)

            # Creating an initial population randomly.
            new_populations = [GARI.initial_population(img_shape=self.target_shape,
                                                       n_individuals=self.sol_per_pop) for _ in range(cores)]

            pool.map_async(self._run_n_generations, new_populations)

            pool.close()
            pool.join()
        else:
            new_population = GARI.initial_population(img_shape=self.target_shape,
                                                     n_individuals=self.sol_per_pop)

            # Display the final generation
            # GARI.show_indivs(new_population, target_shape)

    def _run_n_generations(self, n: int, new_population):
        for generation in range(n):
            fitness_value = GARI.cal_pop_fitness(new_population, model=self.model)
            print(f'Fitness : {numpy.max(fitness_value)}, Iteration : {generation}')

            # Selecting the best parents in the population for mating.
            parents = GARI.select_mating_pool(pop=new_population,
                                              qualities=fitness_value,
                                              num_parents=self.num_parents_mating)

            # Generating next generation using crossover.
            new_population = GARI.crossover(parents,
                                            img_shape=self.target_shape,
                                            n_individuals=self.sol_per_pop)

            new_population = GARI.mutation(population=new_population,
                                           num_parents_mating=self.num_parents_mating,
                                           mut_percent=self.mutation_percent)

            """
            Save best individual in the generation as an image for later visualization.
            """
            GARI.save_images(generation, fitness_value, new_population, self.target_shape,
                             save_point=1000, save_dir=os.environ['CKPT_DIR'])

        return new_population
