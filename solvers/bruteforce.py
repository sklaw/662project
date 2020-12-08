import numpy as np
import random
import itertools
from typing import List
import time
from tqdm import tqdm

# def rotate(x: np.ndarray):
#     for i in range(len(x)):
#         yield np.roll(x, i)



def relaxed_contigious_permute(y: float, c: int, n: int):
    if y == 1:
        for idxs in itertools.combinations(range(n), c):
            x = np.zeros(shape=n)
            x[idxs,] = 1
            yield x
    elif y == 0:
        x = np.zeros(shape=n)
        x[0:c] = 1
        for i in range(n-c+1):
            yield np.roll(x, i)
    else:
        for idxs in itertools.combinations(range(n), c):
            permx = np.zeros(shape=n)
            permx[idxs,] = 1

            breaks = np.abs((np.correlate(permx, np.array([1, 1])) == 2).sum() - c + 1)
            if breaks > y * (c-1):
                continue
            # print(np.abs((np.correlate(permx, np.array([1, 1])) == 2).sum() - c + 1))
            yield np.array(permx)


def bruteforce_solver(A: np.ndarray, iterator: iter, terminate_early: bool, m: int):
    bestobj = 0
    bestx = None

    iter = 0
    starttime = time.time()
    for iter, x in tqdm(enumerate(iterator)):
        x = np.array(x)
        # print(x)
        obj = (A * x).any(1).sum()
        if bestobj < obj:
            bestobj = obj
            bestx = x
        if terminate_early and bestobj == m:  # early termination
            print(f"early termination @ evaluation #{iter + 1}")
            break
    runtime = time.time() - starttime
    return runtime, bestx, bestobj, iter

def main():
    m = 100  # students
    n = 10  # time slots
    c = 5  # meeting slots

    y = 4 / (c-1)  # y(c-1) = maximum number of breaks

    terminate_early = False

    assert c <= n
    assert y <= 1 and y >= 0

    p = 0.7  # probability of a student being available at any particular time
    A = np.random.choice(a=[0, 1], size=(m, n), p=[1-p, p])
    # A = np.array([
    #     [0, 0, 0, 0, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 1, 0, 0, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 1, 1],
    # ])
    print(f"A: {A}")

    # iterate over all possible x and choose the best one
    x = np.zeros(shape=n)
    x[0:c] = 1
    iterator = relaxed_contigious_permute(y, c, n)

    runtime, bestx, bestobj, iterations = bruteforce_solver(A, iterator, terminate_early, m)

    print(f"Evaluation time: {runtime}sec")
    print(f"bestobj: {bestobj}")
    print(f"bestx: {bestx}")
    print(f"objective function evaluations: {iterations + 1}")

if __name__ == '__main__':
    main()
