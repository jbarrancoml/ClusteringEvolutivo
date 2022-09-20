# dividing the datasets into two parts i.e. training datasets and test datasets
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np


def create_clusters_from_labels(labels, num_clusters):
    """
    Create a list of clusters, where each cluster is a list with the indexes of its samples
    :param labels: For each sample, the cluster to which is belongs
    :param num_clusters: Number of clusters (i.e., different label values)
    :return: A list with the samples distributed among clusters according to the labels
    """
    clusters = list()
    if num_clusters > 0 and len(labels) > 0:
        for i in range(0, num_clusters):
            clusters.append(list())
        for i in range(0, len(labels)):
            index = labels[i]
            clusters[index].append(i)
    return clusters


def generate_label_samples_from_clusters(clusters, num_clusters, samples):
    """
    Assign the cluster id as label of each sample in the dataframe. This allows computing performance metrics.
    """
    size = len(samples)
    labels = np.full(size, -1)
    for i in range(0, num_clusters):
        for j in range(0, len(clusters[i])):
            index = clusters[i][j]
            labels[index] = i
    return labels

def main():
    dataset_path = '../data/breast-cancer_bin.csv'

    #load the dataset to a pandas dataFrame
    df = pd.read_csv(dataset_path)

    # Spliting arrays or matrices into random train and test subsets
    from sklearn.model_selection import train_test_split

    #print head
    print(df.columns)

    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # importing random forest classifier from assemble module
    from sklearn.ensemble import RandomForestClassifier
    # creating dataframe of IRIS dataset

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL: ", accuracy_score(y_test, y_pred))

    #Get the false positives list
    false_positives = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_test.values[i]:
            false_positives.append(i)

    print("False positives: ", false_positives)

    #Get the false negatives list
    false_negatives = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_test.values[i]:
            false_negatives.append(i)

    print("False Positives: ", false_positives)

    #Get the list of instances values separated with commas with the indexes of the false positives
    false_positives_values = []
    for i in false_positives:
        false_positives_values.append(list(X_test.values[i]))

    print("False positives values: ", false_positives_values)

    #Get the list of instances with the indexes of the false negatives
    false_negatives_values = []
    for i in false_negatives:
        false_negatives_values.append(list(X_test.values[i]))

    print("False negatives values: ", false_negatives_values)

    import time
    from src.algorithm import evo_clustering

    t_start = time.perf_counter()

    result, labels = evo_clustering(false_positives_values, pop_size=50, max_clusters=3, max_iter=200, show_times=False, show_graphs=True)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    print('Time evoclustering:', total_time)
    print('My silhouette score:', result)

    result = silhouette_score(X=false_positives_values, labels=labels, metric='euclidean')
    print('Sklearn silhouette score:', result)


    #################################
    t_start = time.perf_counter()

    alg = AffinityPropagation(random_state=0, damping=0.7, max_iter=200, convergence_iter=30)
    alg.fit(false_positives_values)
    centroids = alg.cluster_centers_indices_
    num_clusters = len(centroids)
    if len(centroids) == 0:
        clusters, exemplars = None, None
    else:
        clusters = create_clusters_from_labels(alg.labels_, num_clusters)
        exemplars = centroids.tolist()

    t_end = time.perf_counter()
    total_time = t_end - t_start
    print('Alg labels:', type(alg.labels_))
    result = silhouette_score(X=false_positives_values, labels=alg.labels_, metric='euclidean')

    print('Time affinity propagation:', total_time)
    print('Silhouette score:', result)
    ##################################


if __name__ == '__main__':
    main()
