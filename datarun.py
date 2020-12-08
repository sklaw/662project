from solvers.main import mainsolver
from solvers.bruteforce import bruteforce_solver, relaxed_contigious_permute

import numpy as np
import itertools
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)


def main():
    m_a = [50, 100, 200]  # students
    n_a = [50]  # time slots
    c_a = [1, 2, 3, 4]  # meeting slots

    # breaks_a = []
    contigious_a = [False, True]

    terminate_early = False

    p = 0.5  # probability of a student being available at any particular time

    columns = ['m', 'n', 'c', 'contigious', 'A', 'runtime', 'optimal_x', 'optimal_p', 'solver']
    df = pd.DataFrame(columns=columns)

    for m, n, c, contigious in itertools.product(m_a, n_a, c_a, contigious_a):
        print(f"m: {m} n: {n} c: {c} contigious: {contigious}")
        if contigious:
            y = 0
        else:
            y = 1
            # y = breaks / (c - 1)  # y(c-1) = maximum number of breaks

        if not (c <= n):
            continue
        if not (y <= 1 and y >= 0):
            continue

        A = np.random.choice(a=[0, 1], size=(m, n), p=[1-p, p])

        x = np.zeros(shape=n)
        x[0:c] = 1
        iterator = relaxed_contigious_permute(y, c, n)

        # print(f"A: {A}")

        runtime, bestx, bestobj, _ = bruteforce_solver(A, iterator, terminate_early, m)
        df.loc[len(df)] = [m, n, c, contigious, A, runtime, bestx, bestobj, 'bruteforce']

        runtime, bestx, bestobj, _ = mainsolver(A, n, m, c, contigious)
        df.loc[len(df)] = [m, n, c, contigious, A, runtime, bestx, bestobj, 'mainsolver']

    df.to_csv('output_run2.csv')

if __name__ == '__main__':
    main()