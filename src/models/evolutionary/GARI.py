import functools
import itertools
import operator
import random
import os

import matplotlib.pyplot
import numpy as np

import torch
from torchvision.transforms import ToTensor

from utils.train_constants import DEVICE, GA_IMG_SIZE

"""
This work introduces a simple project called GARI (Genetic Algorithm for Reproducing Images).
GARI reproduces a single image using Genetic Algorithm (GA) by evolving pixel values.

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

This project is implemented using Python 3.5 by Ahmed F. Gad.
Contact info:
ahmed.fawzy@ci.menofia.edu.eg
https://www.linkedin.com/in/ahmedfgad/
"""


def img2chromosome(img_arr):
    """
    First step in GA is to represent/encode the input as a sequence of characters.
    The encoding used is value encoding by giving each gene in the 
    chromosome its actual value in the image.
    Image is converted into a chromosome by reshaping it as a single row vector.
    """
    chromosome = np.reshape(a=img_arr,
                            newshape=(functools.reduce(operator.mul,
                                                       img_arr.shape)))
    return chromosome


def initial_population(img_shape, n_individuals=8, face=None):
    """Creating an initial population randomly."""
    population = np.random.random(size=(n_individuals,
                                        functools.reduce(operator.mul, img_shape))) * 2 - 1
    # population[-1, :] = img2chromosome(np.resize(face, new_shape=img_shape))
    # for i, individual in enumerate(population):
    #     img = chromosome2img(individual, img_shape=img_shape)
    #     img[0, IMG_SIZE//5:4*IMG_SIZE//5, IMG_SIZE//5:4*IMG_SIZE//5] = 198/255
    #     img[1, IMG_SIZE//5:4*IMG_SIZE//5, IMG_SIZE//5:4*IMG_SIZE//5] = 134/255
    #     img[2, IMG_SIZE//5:4*IMG_SIZE//5, IMG_SIZE//5:4*IMG_SIZE//5] = 66/255
    #
    #     img[:, IMG_SIZE//3: IMG_SIZE//3 + IMG_SIZE//10, IMG_SIZE//4:3*IMG_SIZE//4] = 0.25
    #     population[i, :] = img2chromosome(img)
    return population


def chromosome2img(chromosome, img_shape):
    """
    First step in GA is to represent the input in a sequence of characters.
    The encoding used is value encoding by giving each gene in the chromosome 
    its actual value.
    """
    img_arr = np.reshape(a=chromosome, newshape=img_shape)
    return img_arr


def fitness_fun_difference(target_chrom, indiv_chrom):
    """
    Calculating the fitness of a single solution.
    The fitness is basicly calculated using the sum of absolute difference
    between genes values in the original and reproduced chromosomes.
    """
    quality = np.mean(np.abs(target_chrom - indiv_chrom))
    quality = -quality
    return quality


def cal_pop_fitness(pop, model):
    """
    This method calculates the fitness of all solutions in the population.
    """
    images = np.reshape(pop, newshape=(pop.shape[0], 3, GA_IMG_SIZE, GA_IMG_SIZE))
    images = torch.from_numpy(images).float().to(DEVICE)
    fitness = model(images)
    return fitness.cpu().detach().numpy()


def select_mating_pool(pop, qualities, num_parents):
    """
    Selects the best individuals in the current generation, according to the 
    number of parents specified, for mating and generating a new better population.
    """
    parents = np.empty((num_parents, pop.shape[1]), dtype=np.float32)
    for parent_num in range(num_parents):
        # Retrieving the best unselected solution.
        max_qual_idx = np.where(qualities == np.max(qualities))
        max_qual_idx = max_qual_idx[0][0]
        # Appending the currently selected 
        parents[parent_num, :] = pop[max_qual_idx, :]
        """
        Set quality of selected individual to a negative value to not get 
        selected again. Algorithm calculations will just make qualities >= 0.
        """
        qualities[max_qual_idx] = -1
    return parents


def crossover(parents, img_shape, n_individuals=8):
    """
    Applying crossover operation to the set of currently selected parents to 
    create a new generation.
    """
    new_population = np.empty(shape=(n_individuals,
                                     functools.reduce(operator.mul, img_shape)),
                              dtype=np.float32)

    """
    Selecting the best previous parents to be individuals in the new generation.

    **Question** Why using the previous parents in the new population?
    It is recommended to use the previous best solutions (parents) in the new 
    generation in addition to the offspring generated from these parents and 
    not use just the offspring.
    The reason is that the offspring may not produce the same fitness values 
    generated by their parents. Offspring may be worse than their parents.
    As a result, if none of the offspring are better, the previous generations 
    winners will be reselected until getting a better offspring.
    """
    # Previous parents (best elements).
    new_population[0:parents.shape[0], :] = parents

    # Getting how many offspring to be generated. If the population size is 8 and number of
    # parents mating is 4, then number of offspring to be generated is 4.
    num_newly_generated = n_individuals - parents.shape[0]
    # Getting all possible permutations of the selected parents.
    parents_permutations = list(itertools.permutations(iterable=np.arange(0, parents.shape[0]), r=2))
    # Randomly selecting the parents permutations to generate the offspring.
    selected_permutations = random.sample(range(len(parents_permutations)),
                                          num_newly_generated)

    comb_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        # Generating the offspring using the permutations previously selected randomly.
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]

        # Applying crossover by exchanging half of the genes between two parents.
        half_size = np.int32(new_population.shape[1] / 2)
        new_population[comb_idx + comb, 0:half_size] = parents[selected_comb[0],
                                                       0:half_size]
        new_population[comb_idx + comb, half_size:] = parents[selected_comb[1],
                                                      half_size:]

    return new_population


def mutation(population, num_parents_mating, mut_percent):
    """
    Applying mutation by selecting a predefined percent of genes randomly.
    Values of the randomly selected genes are changed randomly.
    """
    for idx in range(num_parents_mating, population.shape[0]):
        if np.random.random() < 0.6:
            # A predefined percent of genes are selected randomly.
            rand_idx = np.uint32(np.random.random(size=np.uint32(mut_percent / 100 * population.shape[1]))
                                 * population.shape[1])
            # Changing the values of the selected genes randomly.
            new_values = np.random.random(size=rand_idx.shape[0]) * 2 - 1
            # new_values = np.random.normal(loc=0.0, scale=1.0, size=rand_idx.shape[0])
            # Updating population after mutation.
            population[idx, rand_idx] = new_values
    return population


def save_images(curr_iteration, new_population, model, im_shape,
                save_point, save_dir, log_tag):
    """
    Saving best solution in a given generation as an image in the specified directory.
    Images are saved accoirding to stop points to avoid saving images from 
    all generations as saving mang images will make the algorithm slow.
    """
    qualities = cal_pop_fitness(new_population, model)
    if np.mod(curr_iteration, save_point) == 0:
        # Selecting best solution (chromosome) in the generation.
        best_solution_chrom = new_population[np.where(qualities ==
                                                      np.max(qualities))[0][0], :]
        # Decoding the selected chromosome to return it back as an image.
        best_solution_img = chromosome2img(best_solution_chrom, im_shape)
        # Saving the image in the specified directory.
        best_solution_img = (best_solution_img + 1) / 2
        path = os.path.join(save_dir, f'GA_results_{log_tag}')
        if not os.path.exists(path):
            os.mkdir(path)
        matplotlib.pyplot.imsave(os.path.join(path, f'solution_{curr_iteration}_{np.max(qualities)}.png'),
                                 best_solution_img.transpose(1, 2, 0))
        # matplotlib.pyplot.imshow(best_solution_img, cmap='gray')
        # matplotlib.pyplot.show()


def show_indivs(individuals, im_shape):
    """
    Show all individuals as image in a single graph.
    """
    num_ind = individuals.shape[0]
    fig_row_col = 1
    for k in range(1, np.uint16(individuals.shape[0] / 2)):
        if np.floor(np.power(k, 2) / num_ind) == 1:
            fig_row_col = k
            break
    fig1, axis1 = matplotlib.pyplot.subplots(fig_row_col, fig_row_col)

    curr_ind = 0
    for idx_r in range(fig_row_col):
        for idx_c in range(fig_row_col):
            if curr_ind >= individuals.shape[0]:
                break
            else:
                curr_img = chromosome2img(individuals[curr_ind, :], im_shape)
                axis1[idx_r, idx_c].imshow(curr_img)
                # print(curr_img.min(), curr_img.max())
                curr_ind = curr_ind + 1
    matplotlib.pyplot.show()
