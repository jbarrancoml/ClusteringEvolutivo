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
