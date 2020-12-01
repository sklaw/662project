import numpy as np
import random
import itertools
from typing import List

def rotate(x: np.ndarray):
    for i in range(len(x)):
        yield np.roll(x, i)

def main():
    m = 8  # students
    n = 7  # time slots
    c = 2  # meeting slots
    contigious = True

    assert c <= n

    p = 0.5  # probability of a student being available at any particular time
    A = np.random.choice(a=[0, 1], size=(m, n), p=[1-p, p])

    # choose a random x
    if contigious:
        x = np.zeros(shape=n)
        xsel = random.randint(0, n - c)
        x[xsel:xsel + c] = 1
    else:
        xsel = np.random.choice(a=5, size=c, replace=False)
        x = np.zeros(shape=n)
        x[xsel] = 1
    print(f"A: {A}")
    print(f"x: {x}")

    # evaluate objective function
    obj = (A * x).any(1).sum()

    print(f"obj: {obj}")


    # iterate over all possible x and choose the best one
    x = np.zeros(shape=n)
    x[0:c] = 1
    if contigious:
        iterator = rotate(x)
    else:
        iterator = itertools.permutations(x)

    bestobj = 0
    bestx = None
    for iter, x in enumerate(iterator):
        x = np.array(x)
        obj = (A * x).any(1).sum()
        if bestobj < obj:
            bestobj = obj
            bestx = x
        if bestobj == m:  # early termination
            print(f"early termination @ evaluation #{iter}")
            break

    print(f"bestobj: {bestobj}")
    print(f"bestx: {bestx}")
    print(f"objective function evaluations: {iter}")

if __name__ == '__main__':
    main()
