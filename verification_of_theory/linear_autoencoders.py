import numpy as np
from numpy.linalg import svd, norm, eig
from copy import deepcopy


def forward(A, B, X):
    return A @ B @ X

def loss_fn(A, B, X):
    return np.mean(np.power(X - A @ B @ X, 2))

def step(A, B, X, eta):
    A_ = A + eta * (X - A @ B @ X) @ X.T @ B.T
    B_ = B + eta * A.T @ (X - A @ B @ X) @ X.T
    return A_, B_

def main():
    SEED = 17
    np.random.seed(SEED)
    d = 10
    n = 12
    X = np.random.randn(d, n)


    A0 = np.zeros((d,d))
    B0 = 1/n * X @ X.T
    U, s, Vt = svd(B0)
    T = 500000

    A_, B_ = deepcopy(A0), deepcopy(B0)
    eta = 1e-4

    for t in range(T):
        A_, B_ = step(A_, B_, X, eta)
        loss = loss_fn(A_, B_, X)
        if t % 1000 == 0:
            print("Loss: ", loss)

    Ub, sb, Vtb = svd(B_)

    l = np.sqrt(np.sqrt(np.power(s, 4)/4 + 1) + np.power(s, 2)/2)

    error = np.mean(np.power(l - sb, 2))
    print("Error between predicted singular values of B and actual: ", error)

if __name__ == "__main__":
    main()
