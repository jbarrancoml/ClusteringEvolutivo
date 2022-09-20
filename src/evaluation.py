######################################################################################################################
# File: evaluation.py
# Goal: Class to evaluate a solution (cluster prototypes to explain error samples) based on distances
# Author: Aurora Ramirez (UCO)
# Version: 1
# Date: April, 2021
#######################################################################################################################

# TODO

# This is just a test to see if this approach works, not the class

import random
import numpy as np
from sklearn.metrics import silhouette_score


def random_problem(size, min_coord=0.0, max_coord=10.0):
    problem = list(range(size))
    for i in range(size):
        x = round(random.uniform(min_coord, max_coord), 4)
        y = round(random.uniform(min_coord, max_coord), 4)
        problem[i] = [x, y]
    return problem


def silhouette_sklearn(dataset, labels):
    result = silhouette_score(X=dataset, labels=labels, metric='euclidean')
    return result


def silhouette_coef_eval(solution, distance_matrix, verbose=False):
    # a: The mean distance between a sample and all other points in the same class.
    # b: The mean distance between a sample and all other points in the next nearest cluster.
    clusters = solution.get_clusters()
    silhouette_coef = 0
    ### Get mean distance between points in same class ###
    a_score = 0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                a_score += distance_matrix[i][j]
    #a_score = mean_distance_between_points_in_same_class(clusters, n_clusters)
    #b_score = mean_distance_between_points_in_next_cluster(clusters, n_clusters)

    b_score = 0

    for cluster in clusters:
        distance = 99999999
        lowest_distance = distance
        nearest_cluster = 0
        n_cluster = 0
        cluster_data = []
        for cluster_ in clusters:
            if cluster_ != cluster:
                distance = distance_matrix[cluster_[0]][cluster[0]]
                if distance < lowest_distance:
                    nearest_cluster = n_cluster
            cluster_data.append(nearest_cluster)

        for i in range(len(cluster)):
            for j in range(len(clusters[cluster_data[n_cluster]])):
                b_score += distance_matrix[i][j]
        n_cluster += 1

    silhouette_coef = (b_score - a_score) / max(a_score, b_score)

    return silhouette_coef

def mean_distance_between_points_in_next_cluster(clusters, n_clusters):
    # initialize a_score with length n_clusters with 0
    b_score = [0] * n_clusters
    other_clusters = [] * (n_clusters - 1)
    for i in range(len(clusters)):

        # sum the clusters that are not the current cluster
        for k in range(len(clusters)):
            if k != i:
                other_clusters += clusters[k]

        if len(clusters[i]) > 0:
            b_score[i] += calculate_distance_between_many_points_to_many_points(clusters[i], other_clusters)

    return sum(b_score) / len(clusters)


def mean_distance_between_points_in_same_class(clusters, n_clusters):
    # initialize a_score with length n_clusters with 0
    a_score = [0] * n_clusters

    for i in range(len(clusters)):
        if len(clusters[i]) > 0:
            a_score[i] += calculate_distance_between_many_points(clusters[i])

    return sum(a_score) / len(clusters)


def calculate_distance_between_two_points(point1, point2):
    return pow(np.linalg.norm(np.array(point1) - np.array(point2)), 2)


def calculate_distance_between_many_points(points):
    distance = 0
    for i in range(len(points)):
        for k in range(len(points)):
            distance += calculate_distance_between_two_points(points[i], points[k])
    return distance


def calculate_distance_between_many_points_to_many_points(points1, points2):
    distance = 0
    for i in range(len(points1)):
        for k in range(len(points2)):
            distance += calculate_distance_between_two_points(points1[i], points2[k])
    return distance


def translate_solution(clusters, samples):
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            clusters[i][j] = samples[clusters[i][j]]
    pass


def main():
    problem = random_problem(5)
    cluster1, cluster2 = [], []
    solution = []

    print('===== Problem Generated =====')
    print(problem)

    for i in range(len(problem)):
        if random.randint(1, 2) == 1:
            cluster1.append(problem[i])
        else:
            cluster2.append(problem[i])

    print('\n===== Silhouette Coefficient 1 =====')
    cluster1 = [1, 1.3, 1.1, 0.8]
    cluster2 = [5.2, 5.7, 5.3, 4.5, 4.1, 4.0]

    silhouette_coef_eval([cluster1, cluster2], 2)
    print('Cluster 1:', cluster1)
    print('Cluster 2:', cluster2)

    print('\n===== Silhouette Coefficient 2 =====')
    cluster3 = [1, 5.7, 4.0, 1.1, 5.2]
    cluster4 = [1.3, 0.8, 5.3, 4.5, 4.1]

    silhouette_coef_eval([cluster3, cluster4], 2)

    print('\n===== Silhouette Coefficient 3 =====')
    cluster5 = [1, 1, 1, 1, 1]
    cluster6 = [10, 10, 10, 10, 10]

    silhouette_coef_eval([cluster5, cluster6], 2)

    print('\n===== Silhouette Coefficient 4 =====')
    cluster7 = [1]
    cluster8 = [1.2, 1.5, 2.5, 2.6, 2.7]

    silhouette_coef_eval([cluster7, cluster8], 2)


if __name__ == '__main__':
    main()
