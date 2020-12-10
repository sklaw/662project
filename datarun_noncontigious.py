from solvers.main import mainsolver
from solvers.bruteforce import bruteforce_solver, relaxed_contigious_permute
from solvers.noncontiguous_penality import noncpsolver
from solvers.noncontiguous_soft_constraints import noncspsolver

import numpy as np
import itertools
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)


def main():
    m_a = [50]  # students
    n_a = [50]  # time slots
    c_a = [1, 2, 3, 4, 5, 6, 7]  # meeting slots

    breaks_a = [2]  # total number of breaks

    penalty_a = [8, 10, 15]  # soft constraint parameter
    L_a = [1, 2, 5]  # max number of zeros

    p = 0.5  # probability of a student being available at any particular time

    columns = ['m', 'n', 'c', 'contigious', 'A', 'runtime', 'optimal_x', 'optimal_p', 'solver',
               'breaks', 'L', 'penalty']
    df = pd.DataFrame(columns=columns)

    # noncontigious soft constraints
    for m, n, c, breaks, L in itertools.product(m_a, n_a, c_a, breaks_a, L_a):
        print(f"m: {m} n: {n} c: {c} breaks: {breaks} L: {L}")

        if not (c <= n):
            continue
        if not (L <= n-2 and L >= 1):
            continue
        if not (breaks < c):
            continue

        A = np.random.choice(a=[0, 1], size=(m, n), p=[1-p, p])

        runtime, bestx, bestobj, _ = noncspsolver(A, n, m, c, breaks, L)
        df.loc[len(df)] = [m, n, c, None, A, runtime, bestx, bestobj, 'noncspsolver', breaks, L, None]

    # noncontigious constraints
    for m, n, c, breaks, p1 in itertools.product(m_a, n_a, c_a, breaks_a, penalty_a):
        print(f"m: {m} n: {n} c: {c} breaks: {breaks} penalty: {p1}")

        if not (c <= n):
            continue
        if not (breaks < c):
            continue

        A = np.random.choice(a=[0, 1], size=(m, n), p=[1-p, p])

        runtime, bestx, bestobj, _ = noncpsolver(A, n, m, c, breaks, p1)
        df.loc[len(df)] = [m, n, c, None, A, runtime, bestx, bestobj, 'noncpsolver', breaks, None, p1]


    df.to_csv('output_noncontigious.csv')

if __name__ == '__main__':
    main()