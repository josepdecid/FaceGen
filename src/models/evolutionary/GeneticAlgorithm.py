import os
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
from models.evolutionary import GARI
from skimage import io
from skimage.filters import gaussian

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


def run_genetic_algorithm(model: FaceClassifier):
    # Reading target image to be reproduced using Genetic Algorithm (GA).
    target_im = io.imread('face.jpg')

    # Target image after enconding. Value encoding is used.
    target_chromosome = GARI.img2chromosome(target_im)

    # Population size
    sol_per_pop = 75
    # Mating pool size
    num_parents_mating = 10
    # Mutation percentage
    mutation_percent = 0.05
    # Iterations
    iterations = 10000

    """
    There might be inconsistency between the number of selected mating parents and 
    number of selected individuals within the population.
    In some cases, the number of mating parents are not sufficient to 
    reproduce a new generation. If that occurred, the program will stop.
    """
    num_possible_permutations = len(list(itertools.permutations(
        iterable=numpy.arange(0, num_parents_mating), r=2)))
    num_required_permutations = sol_per_pop - num_possible_permutations
    if num_required_permutations > num_possible_permutations:
        print(
            "\n*Inconsistency in the selected populatiton size or number of parents.*"
            "\nImpossible to meet these criteria.\n"
        )
        sys.exit(1)

    # Creating an initial population randomly.
    new_population = GARI.initial_population(img_shape=target_im.shape,
                                             n_individuals=sol_per_pop)

    for iteration in range(iterations + 1):
        # Measing the fitness of each chromosome in the population.
        qualities = GARI.cal_pop_fitness(new_population, model=model)
        print('Quality : ', numpy.max(qualities), ', Iteration : ', iteration)

        # Selecting the best parents in the population for mating.
        parents = GARI.select_mating_pool(new_population, qualities,
                                          num_parents_mating)

        # Generating next generation using crossover.
        new_population = GARI.crossover(parents, target_im.shape,
                                        n_individuals=sol_per_pop)

        """
        Applying mutation for offspring.
        Mutation is important to avoid local maxima. Avoiding mutation makes 
        the GA falls into local maxima.
        Also mutation is important as it adds some little changes to the offspring. 
        If the previous parents have some common degaradation, mutation can fix it.
        Increasing mutation percentage will degarde next generations.
        """
        new_population = GARI.mutation(population=new_population,
                                       num_parents_mating=num_parents_mating,
                                       mut_percent=mutation_percent)

        # if iteration % 500 == 0:
        #     for i, individual in enumerate(new_population):
        #         img = GARI.chromosome2img(individual, img_shape=(200, 200, 3))
        #         img = gaussian(img, multichannel=True, preserve_range=True)
        #         new_population[i] = GARI.img2chromosome(numpy.array(img))

        """
        Save best individual in the generation as an image for later visualization.
        """
        GARI.save_images(iteration, qualities, new_population, target_im.shape,
                         save_point=1000, save_dir=os.environ['CKPT_DIR'])

    # Display the final generation
    GARI.show_indivs(new_population, target_im.shape)
