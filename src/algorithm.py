######################################################################################################################
# File: algorithm.py
# Goal: Class to run the evolutionary algorithm for optimal selection of cluster prototypes
# Author: Aurora Ramirez (UCO)
# Version: 1
# Date: April, 2021
#######################################################################################################################

# TODO

from random import Random
from src.initialization import PopulationCreator
from src.repair import RepairOperator
from src.selection import ParentSelector
from src.crossover import Crossover
from src.mutation import Mutation
from src.evaluation import silhouette_coef_eval, silhouette_sklearn
import numpy as np
import time
# from matplotlib import pyplot as plt
import plotly.graph_objects as go


def compute_distance_matrix(dataset, genotype_length):
    """
    Compute the distance matrix between the points in the genotype with the dataset
    :param dataset: Array with the points in the dataset
    :return: Distance matrix
    """
    distance_matrix = np.zeros((genotype_length, genotype_length))
    for i in range(0, genotype_length):
        for j in range(0, genotype_length):
            distance_sum = 0
            if isinstance(dataset[i], list):
                for k in range(0, len(dataset[i])):
                    distance_sum += get_distance_c(dataset[i][k], dataset[j][k])
            else:
                distance_sum = get_distance(i, j, dataset)
            distance_matrix[i][j] = distance_sum
    return distance_matrix


def get_distance(i, j, dataset):
    """
    Calculate the distance between two points in the dataset
    :param i: Index of the first point
    :param j: Index of the second point
    :return: Distance between the two points
    """
    return np.linalg.norm(dataset[i] - dataset[j])


def get_distance_b(i, j, dataset_a, dataset_b):
    """
    Calculate the distance between two points in the dataset
    :param i: Index of the first point
    :param j: Index of the second point
    :return: Distance between the two points
    """
    distance_sum = 1
    if len(dataset_a[i]) > 1:
        for k in len(dataset_a[i]):
            distance_sum += get_distance(dataset_a[i][k], dataset_b[i][k])

    else:
        distance_sum = np.linalg.norm(dataset_a[i] - dataset_b[j])
    return distance_sum


def get_distance_c(i, j):
    return np.linalg.norm(i - j)


def get_clusters(solution, max_clusters):
    clusters = [0] * max_clusters
    for i in range(len(clusters)):
        clusters[i] = []
    genotype = solution.get_genotype()
    for i in range(0, len(genotype)):
        clusters[genotype[i]].append(int(i / 2))
    return clusters


def plot_clusters(clusters):
    clusters_x = []
    clusters_y = []

    for cluster in clusters:
        cluster_x = []
        cluster_y = []
        for point in cluster:
            cluster_x.append(point[0])
            cluster_y.append(point[1])
        clusters_x.append(cluster_x)
        clusters_y.append(cluster_y)

    plots = []
    colors = ['#e33e3e', '#583b4d', '#64ff00', '#69ffcb', '#17f1ff', '#0500ab', '#ff009e']
    for i in range(len(clusters_x)):
        plots.append(go.Scatter(y=clusters_y[i], x=clusters_x[i], name='Cluster' + str(i + 1), mode='markers',
                                marker={'color': colors[i], 'size': 24}))
    layout = go.Layout(title=go.layout.Title(text="Clustering", x=0.5),
                       yaxis_title="C1",
                       xaxis_title="C2",
                       )
    fig = go.Figure(data=plots, layout=layout)
    fig.show()


def plot_best_fitness_graph(best_fitness_array, epochs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best_fitness_array, x=epochs, name='best fitness each epoch'))
    fig.update_layout(title="Best fitness each epoch",
                      yaxis_title="fitness",
                      xaxis_title="epoch",
                      )
    fig.show()


def plot_mean_fitness_graph(mean_fitness_array, epochs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mean_fitness_array, x=epochs, name='mean fitness each epoch'))
    fig.update_layout(title="Mean fitness each epoch",
                      yaxis_title="fitness",
                      xaxis_title="epoch",
                      )
    fig.show()


def evo_clustering(dataset, pop_size=250, min_clusters=2, max_clusters=5, max_iter=100, show_times=False, show_graphs=False):
    rnd = Random()
    rnd.seed(0)
    num_samples = len(dataset)
    distance_matrix = compute_distance_matrix(dataset, num_samples)
    creator = PopulationCreator(pop_size, num_samples, max_clusters, distance_matrix, rnd)
    population = creator.create_population()
    print('Distance matrix:', distance_matrix)
    # print('Population')
    # print(population.print_population())

    best_fitness = -1
    best_solution = population.get_solutions()[0]

    best_fitness_each_epoch = []
    mean_fitness_each_epoch = []
    epochs = []

    for iter in range(max_iter):
        print('Progress:', (iter / max_iter) * 100, '%')
        # Random evaluation

        time_start = time.time()

        mean_fitness = 0
        fitness_sum = 0

        for s in population.get_solutions():
            labels = s.get_point_indexes(dataset)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print('n_clusters:', n_clusters)
            if n_clusters != 1:
                fitness = silhouette_sklearn(dataset, labels)
            else:
                fitness = -1
            print(fitness)
            fitness_sum += fitness
            s.set_fitness(fitness)
            if best_fitness < fitness:
                best_fitness = fitness
                best_solution = s

        mean_fitness = fitness_sum / pop_size

        time_stop = time.time()
        time_eval = time_stop - time_start

        time_start = time.time()

        selector = ParentSelector(rnd, tournament_size=2)
        parents = selector.tournament_selection(population.get_solutions())

        time_stop = time.time()
        time_select = time_stop - time_start

        time_start = time.time()

        crossover_op = Crossover(rnd, probability=0.7)
        offspring = crossover_op.recombine_parents(parents)

        time_stop = time.time()
        time_crossover = time_stop - time_start

        time_start = time.time()
        mutation_op = Mutation(rnd, genotype_length=num_samples * 2, mut_probability=0.3)
        mutants = mutation_op.mutate_solutions(offspring)
        time_stop = time.time()
        time_mutation = time_stop - time_start

        repair_op = RepairOperator(rnd)
        repaired_population = repair_op.repair_population(mutants)

        if iter % 5 == 0:
            print('current best fitness:', best_fitness)
        population.set_population(repaired_population)

        ##### TIME DASHBOARD #####
        if show_times:
            print('Time evaluation:', time_eval)
            print('Time selection:', time_select)
            print('Time crossover:', time_crossover)
            print('Time mutation:', time_mutation)
            print('Time total:', time_eval + time_select + time_crossover + time_mutation)

        best_fitness_each_epoch.append(best_solution.get_fitness())
        mean_fitness_each_epoch.append(mean_fitness)
        epochs.append(iter)

    sbest_phenotype = best_solution.solution_to_eval(dataset)

    if show_graphs:
        plot_clusters(sbest_phenotype)
        plot_best_fitness_graph(best_fitness_each_epoch, epochs)
        plot_mean_fitness_graph(mean_fitness_each_epoch, epochs)

    if show_info:
        print('Best fitness achieved:', best_fitness)
        print('Best solution:', best_solution.get_point_indexes(dataset))

    return best_fitness, best_solution.get_point_indexes(dataset)
    # return clusters, centroids
