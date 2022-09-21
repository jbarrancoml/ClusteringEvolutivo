######################################################################################################################
# File: constraints.py
# Goal: Class check solution constraints and repair them
# Author: Aurora Ramirez (UCO)
# Version: 1
# Date: April, 2021
#######################################################################################################################

# TODO
import numpy
import numpy as np

from src.solution import Solution


class RepairOperator:
    def __init__(self, rnd):
        self.rnd = rnd

    def repair_population(self, population):
        new_population = []
        for s in population:
            repaired_solution = self.check_genotype(s)
            new_population.append(repaired_solution)
        return new_population

    def check_genotype(self, solution):
        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        max_clusters = solution.get_max_clusters()
        min_clusters = 2
        repaired_genotype = solution.get_genotype().copy()

        n_clusters = solution.get_number_clusters()
        if 2 > n_clusters:
            repaired_genotype = self.repair_underflow(solution.get_genotype(), min_clusters)

        elif n_clusters > 7:
            repaired_genotype = self.repair_overflow(solution, max_clusters)

        solution.set_genotype(repaired_genotype)

        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        return solution

    def repair_overflow(self, solution, max_clusters):
        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        assert max_clusters is not None
        repaired_genotype = [0]*len(solution.get_genotype())

        medoid_indexes = solution.get_medoid_indexes()
        
        self.rnd.shuffle(medoid_indexes)
        medoid_indexes = medoid_indexes[:max_clusters]

        for i in medoid_indexes:
            repaired_genotype[i] = 1

        repaired_genotype = np.array(repaired_genotype)

        assert isinstance(repaired_genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'
        return repaired_genotype

    def repair_underflow(self, genotype, min_clusters):
        assert isinstance(genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'

        repaired_genotype = [0]*len(genotype)
        posible_indexes = list(range(0, len(genotype)-1))

        for i in range(min_clusters):
            random_index = self.rnd.choice(posible_indexes)
            repaired_genotype[random_index] = 1
            posible_indexes.remove(random_index)

        repaired_genotype = np.array(repaired_genotype)

        assert isinstance(repaired_genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'
        return repaired_genotype
