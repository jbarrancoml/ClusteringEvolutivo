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

    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # importing random forest classifier from assemble module
    from sklearn.ensemble import RandomForestClassifier
    # creating dataframe of IRIS dataset

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0, min_samples_split=2,
                                 min_samples_leaf=1, max_features='auto')

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
    from utils.stat_writer import StatWriter

    Conf_evo = {
        'pop_size': [20, 50, 100],
        'max_clusters': [2, 3, 5],
        'max_iterations': [20, 50, 100],
    }

    writer = StatWriter('../results/breast_cancer_rf_bin_EVO')

    writer.add_header(['Silhouette (max)', 'Num Clusters', 'Time'], [str, int, float])

    for pop_size in Conf_evo['pop_size']:
        for max_clusters in Conf_evo['max_clusters']:
            for max_iterations in Conf_evo['max_iterations']:
                start_time = time.time()
                print("Evo Clustering with pop_size: {}, max_clusters: {}, max_iterations: {}".format(pop_size, max_clusters, max_iterations))
                sil_evo, n_clusters = evo_clustering(dataset=false_positives_values, max_clusters=max_clusters,
                                                     pop_size=pop_size, max_iter=max_iterations)
                elapsed_time = time.time() - start_time
                writer.add_row([sil_evo, n_clusters, elapsed_time])
    writer.write_csv_file()
    writer.generate_excel_file()
    writer.close()

    #################################

    Conf_affinity = {
        'd': [0.5, 0.7, 0.9],
        'm': [100, 200, 500],
        'c': [5, 15, 30]
    }

    writer = StatWriter('../results/breast_cancer_rf_bin_AF')

    writer.add_header(['Silhouette (max)', 'Num Clusters', 'Time'], [float, int, float])

    # Affinity Propagation

    for d in Conf_affinity['d']:
        for m in Conf_affinity['m']:
            for c in Conf_affinity['c']:
                start_time = time.time()
                print("Affinity Propagation with d: {}, m: {}, c: {}".format(d, m, c))
                af = AffinityPropagation(damping=d, max_iter=m, convergence_iter=c, random_state=0).fit(false_positives_values)
                elapsed_time = time.time() - start_time
                cluster_centers_indices = af.cluster_centers_indices_
                labels = af.labels_
                n_clusters_ = len(cluster_centers_indices)
                if n_clusters_ > 1:
                    sil_af = silhouette_score(false_positives_values, labels, metric='euclidean')
                else:
                    sil_af = -1.0
                writer.add_row([sil_af, n_clusters_, elapsed_time])

    writer.write_csv_file()
    writer.generate_excel_file()
    writer.close()

    ##################################


if __name__ == '__main__':
    main()
