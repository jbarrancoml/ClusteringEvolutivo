######################################################################################################################
# File: crossover.py
# Goal: Class to create offspring from parent solutions
# Author: Aurora Ramirez (UCO)
# Version: 1
# Date: April, 2021
#######################################################################################################################

from src.solution import Solution
from src.repair import RepairOperator


class Crossover:
    """
    A class to implement crossover operations
    """

    def __init__(self, rnd, probability=0.8):
        """
        Create a crossover operator
        :param rnd: The random number generator
        :param probability: Crossover probability, 0.8 by default
        """
        self.rnd = rnd
        self.probability = probability

    def recombine_parents(self, parents):
        """
        Recombine each pair of parents to generate a pair of offspring
        :param parents: List of parent solutions
        :return: List of offspring (same size)
        """
        offspring = list()
        repair_op = RepairOperator(self.rnd)

        for i in range(0, len(parents)-1, 2):
            parent_1 = parents[i]
            parent_1_genotype = parent_1.get_genotype()
            parent_2 = parents[i + 1]
            parent_2_genotype = parent_2.get_genotype()

            # Generate offspring with a given probability
            if self.rnd.random() > self.probability:
                son_1_genotype, son_2_genotype = self.two_point_crossover(parent_1_genotype,
                                                                          parent_2_genotype)  # He cambiado el crossover de un punto por el de dos

                """# Check constraints and repair solutions, if needed
                son_1_genotype = repair_op.repair_genotype(son_1_genotype, parent_1_genotype)
                son_2_genotype = repair_op.repair_genotype(son_2_genotype, parent_2_genotype)"""

                offspring_1 = Solution(son_1_genotype, parent_1.get_max_clusters(), parent_1.distance_matrix)
                offspring_2 = Solution(son_2_genotype, parent_2.get_max_clusters(), parent_2.distance_matrix)

            # No crossover happens, just copy the parents
            else:
                offspring_1 = parent_1.copy()
                offspring_2 = parent_2.copy()
            offspring.append(offspring_1)
            offspring.append(offspring_2)
        return offspring

    def one_point_crossover(self, parent_genotype_1, parent_genotype_2):
        """
        Perform one-point crossover. The cross point is an even position (cluster id).
        :param parent_genotype_1: Genotype of the first parent
        :param parent_genotype_2: Genotype of the second parent
        :return: Two offspring genotypes combining genes from the parents
        """
        genotype_length = len(parent_genotype_1)

        # Random point to split, but this should be a even position
        point = self.rnd.randrange(0, genotype_length)

        # Copy the genotype part from the corresponding parent
        genotype_1 = parent_genotype_1.copy()
        genotype_1[point:] = parent_genotype_2[point:]

        genotype_2 = parent_genotype_2.copy()
        genotype_2[:point] = parent_genotype_1[:point]

        return genotype_1, genotype_2

    def two_point_crossover(self, parent_genotype_1, parent_genotype_2):  # TODO
        """
        Perform two-point crossover. Both cross point are an even position (cluster id).
        :param parent_genotype_1: Genotype of the first parent
        :param parent_genotype_2: Genotype of the second parent
        :return: Two offspring genotypes combining genes from the parents
        """
        genotype_length = len(parent_genotype_1)

        # A list to choose which indexes are appropriate for cross-point crossover
        availables = []

        # The first random point to split, but this should be a even position
        point1 = self.rnd.randrange(0, genotype_length)

        # At least, there should be one 
        availables = list(range(0, point1 - 2, 2)) + list(range(point1 + 3, genotype_length, 2))
        point2 = self.rnd.choice(availables)

        genotype_1 = parent_genotype_1.copy()
        genotype_1[point1:point2] = parent_genotype_2[point1:point2]

        genotype_2 = parent_genotype_2.copy()
        genotype_2[:point1] = parent_genotype_1[:point1]
        genotype_2[point2:] = parent_genotype_1[point2:]

        """# Genotype 1 Repair

        i = 0
        n = len(genotype_1)
        genotype_1_n_clusters = 0
        while i < n:
            if genotype_1[i] == 1:
                genotype_1_n_clusters += 1
            i = i + 1

        medoid_indexes = []
        for i in range(len(genotype_1)):
            if genotype_1[i] == 1:
                medoid_indexes.append(i)

        while genotype_1_n_clusters > 7:
            genotype_1[self.rnd.choice(medoid_indexes)] = 0
            genotype_1_n_clusters -= 1

        # Genotype 2 Repair

        i = 0
        n = len(genotype_2)
        genotype_2_n_clusters = 0
        while i < n:
            if genotype_2[i] == 1:
                genotype_2_n_clusters += 1
            i = i + 1

        medoid_indexes = []
        for i in range(len(genotype_2)):
            if genotype_2[i] == 1:
                medoid_indexes.append(i)

        while genotype_2_n_clusters > 7:
            genotype_2[self.rnd.choice(medoid_indexes)] = 0
            genotype_2_n_clusters -= 1"""

        return genotype_1, genotype_2
