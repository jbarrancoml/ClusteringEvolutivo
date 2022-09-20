from algorithm import evo_clustering
import random


def random_problem(rnd, n_samples, sample_dim=1):
    if sample_dim > 1:
        problem = []
        for i in range(n_samples):
            sample = []
            for k in range(sample_dim):
                sample.append(rnd.uniform(0, 10))
            problem.append(sample)
    else:
        problem = []
        for i in range(n_samples):
            problem.append(rnd.uniform(0, 10))

    return problem


def main():
    rnd = random.Random()
    problem = random_problem(rnd, 100, sample_dim=2)

    print('===== Problem Generated =====')
    print(problem)

    evo_clustering(problem, pop_size=50, max_clusters=4, max_iter=200, show_times=False)


if __name__ == '__main__':
    main()
