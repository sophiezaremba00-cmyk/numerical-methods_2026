import random
import matplotlib.pyplot as plt

N = 100
EPSILON = 1e-14

def generate_matrix(n):
    A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        A[i][i] += n

    return A


def write_matrix(filename, A):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")


def read_matrix(filename):
    with open(filename, "r") as f:
        return [list(map(float, line.split())) for line in f]


def write_vector(filename, v):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, v)))


def read_vector(filename):
    with open(filename, "r") as f:
        return list(map(float, f.readline().split()))

def matrix_vector_mult(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(A))) for i in range(len(A))]


def vector_norm(v):
    return max(abs(x) for x in v)

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            s = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - s

        if abs(L[k][k]) < 1e-12:
            raise ValueError("LU-розклад неможливий")

        for j in range(k + 1, n):
            s = sum(L[k][i] * U[i][j] for i in range(k))
            U[k][j] = (A[k][j] - s) / L[k][k]

    return L, U

def forward_substitution(L, B):
    Z = []
    for i in range(len(L)):
        s = sum(L[i][j] * Z[j] for j in range(i))
        Z.append((B[i] - s) / L[i][i])
    return Z


def backward_substitution(U, Z):
    n = len(U)
    X = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s
    return X


def solve_lu(L, U, B):
    return backward_substitution(U, forward_substitution(L, B))

def compute_error(A, X, B):
    return max(abs(sum(A[i][j] * X[j] for j in range(len(A))) - B[i]) for i in range(len(A)))

def iterative_refinement(A, L, U, B, X0):
    X = X0[:]
    MAX_ITERS = 10

    AX = matrix_vector_mult(A, X)
    R = [B[i] - AX[i] for i in range(len(B))]

    error_history = [vector_norm(R)]
    delta_history = []

    iterations = 0

    for _ in range(MAX_ITERS):
        delta_X = solve_lu(L, U, R)
        delta_norm = vector_norm(delta_X)

        delta_history.append(delta_norm)

        X = [X[i] + delta_X[i] for i in range(len(X))]

        AX = matrix_vector_mult(A, X)
        R = [B[i] - AX[i] for i in range(len(B))]
        residual_norm = vector_norm(R)

        error_history.append(residual_norm)

        iterations += 1

        if delta_norm <= EPSILON and residual_norm <= EPSILON:
            break

    return X, iterations, error_history, delta_history

def main():

    A = generate_matrix(N)
    write_matrix("A.txt", A)

    X_true = [2.5] * N
    B = matrix_vector_mult(A, X_true)
    write_vector("B.txt", B)

    A = read_matrix("A.txt")
    B = read_vector("B.txt")

    L, U = lu_decomposition(A)

    X = solve_lu(L, U, B)

    error_before = compute_error(A, X, B)

    X_refined, iters, err_hist, delta_hist = iterative_refinement(A, L, U, B, X)

    error_after = compute_error(A, X_refined, B)

    print("Похибка ДО:", error_before)
    print("Похибка ПІСЛЯ:", error_after)
    print("Ітерацій:", iters)

    plt.figure()
    plt.plot(err_hist, marker='o')
    plt.yscale("log")
    plt.title("Падіння нев'язки ||AX - B||")
    plt.xlabel("Ітерація")
    plt.ylabel("Норма")
    plt.grid()

    plt.figure()
    plt.plot(delta_hist, marker='o')
    plt.yscale("log")
    plt.title("Падіння ||ΔX||")
    plt.xlabel("Ітерація")
    plt.ylabel("Норма")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()