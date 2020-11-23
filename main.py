import cvxpy as cp
import numpy as np


if __name__ == "__main__":
    A = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1],
    ])
    c = 2

    m, n = A.shape
    x = cp.Variable(n, boolean=True)
    Y = cp.Variable([m, n])
    Z = cp.Variable([m, n])
    constraints = []

    for i in range(m):
        for j in range(n):
            constraints.append(Y[i, j] >= A[i, j] + x[j] - 1)
            constraints.append(Y[i, j] <= A[i, j])
            constraints.append(Y[i, j] <= x[j])
            constraints.append(Y[i, j] <= 1)
            constraints.append(Y[i, j] >= 0)

        constraints.append(Z[i,0] == Y[i,0])
        for j in range(1, n):
            constraints.append(Z[i, j] <= Z[i, j-1] + Y[i, j])
            constraints.append(Z[i, j] >= Z[i, j-1])
            constraints.append(Z[i, j] >= Y[i, j])
            constraints.append(Z[i, j] <= 1)
            constraints.append(Z[i, j] >= 0)

        constraints.append(x[i] >= 0)
        constraints.append(x[i] <= 1)

    constraints.append(np.ones(n) @ x == c)

    # Contiguous constraint
    y_1 = cp.Variable(n-1)
    y_2 = cp.Variable(n-1)
    y_3 = cp.Variable(n-1)
    for i in range(n-1):
        constraints.append(y_1[i] == x[i])
        constraints.append(y_2[i] == x[i+1])
        # y_3 = y_1 AND y_2
        constraints.append(y_3[i] >= y_1[i] + y_2[i] - 1)
        constraints.append(y_3[i] <= y_1[i])
        constraints.append(y_3[i] <= y_2[i])
        constraints.append(y_3[i] <= 1)
        constraints.append(y_3[i] >= 0)
    constraints.append(np.ones(n-1)@y_3 == np.ones(n)@x-1)


    prob = cp.Problem(cp.Maximize(np.ones(m) @ Z[:,n-1]), constraints)

    prob.solve()

    print("prob.value=", prob.value)
    print("x.value=", x.value)
    print("Y.value=", Y.value)
    print("Z.value=", Z.value)