import random
import matplotlib.pyplot as plt

def generate_matrix(n):
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = row_sum + random.uniform(1, 5)

    return A


def mat_vec_mult(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(A))) for i in range(len(A))]


def vector_norm(v):
    return max(abs(x) for x in v)


def matrix_norm_inf(A):
    return max(sum(abs(A[i][j]) for j in range(len(A))) for i in range(len(A)))


def residual(A, x, b):
    Ax = mat_vec_mult(A, x)
    return vector_norm([Ax[i] - b[i] for i in range(len(A))])


def simple_iteration(A, b, x0, eps, x_true):
    n = len(A)
    x = x0[:]

    tau = 0.9 * 2 / matrix_norm_inf(A)

    errors = []

    for k in range(10000):
        x_new = [
            x[i] - tau * (sum(A[i][j] * x[j] for j in range(n)) - b[i])
            for i in range(n)
        ]

        err = vector_norm([x_new[i] - x_true[i] for i in range(n)])
        res = residual(A, x_new, b)
        errors.append(err)

        if vector_norm([x_new[i] - x[i] for i in range(n)]) < eps or res < eps:
            return x_new, k + 1, errors

        x = x_new

    return x, 10000, errors


def jacobi(A, b, x0, eps, x_true):
    n = len(A)
    x = x0[:]

    errors = []

    for k in range(10000):
        x_new = [0] * n

        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        err = vector_norm([x_new[i] - x_true[i] for i in range(n)])
        res = residual(A, x_new, b)
        errors.append(err)

        if vector_norm([x_new[i] - x[i] for i in range(n)]) < eps or res < eps:
            return x_new, k + 1, errors

        x = x_new

    return x, 10000, errors


def gauss_seidel(A, b, x0, eps, x_true):
    n = len(A)
    x = x0[:]

    errors = []

    for k in range(10000):
        x_new = x[:]

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        err = vector_norm([x_new[i] - x_true[i] for i in range(n)])
        res = residual(A, x_new, b)
        errors.append(err)

        if vector_norm([x_new[i] - x[i] for i in range(n)]) < eps or res < eps:
            return x_new, k + 1, errors

        x = x_new

    return x, 10000, errors


def print_matrix(A, limit=5):
    print(f"Матриця A (перші {limit}x{limit} елементів):")
    for row in A[:limit]:
        print(["{:.3f}".format(x) for x in row[:limit]])


def print_vector(v, name, limit=10):
    print(f"\nВектор {name} (перші {limit} елементів):")
    print(["{:.3f}".format(x) for x in v[:limit]])


def main():
    n = 100
    eps = 1e-14

    A = generate_matrix(n)
    x_true = [2.5] * n
    b = mat_vec_mult(A, x_true)

    print_matrix(A, limit=5)
    print_vector(x_true, "x_true")
    print_vector(b, "b")

    x0 = [1.0] * n

    x_si, it_si, err_si = simple_iteration(A, b, x0, eps, x_true)
    x_j, it_j, err_j = jacobi(A, b, x0, eps, x_true)
    x_gs, it_gs, err_gs = gauss_seidel(A, b, x0, eps, x_true)

    print("\nПроста ітерація:", it_si, "ітерацій")
    print("Похибка:", vector_norm([x_si[i] - 2.5 for i in range(n)]))

    print("\nЯкобі:", it_j, "ітерацій")
    print("Похибка:", vector_norm([x_j[i] - 2.5 for i in range(n)]))

    print("\nЗейдель:", it_gs, "ітерацій")
    print("Похибка:", vector_norm([x_gs[i] - 2.5 for i in range(n)]))

    plt.figure(figsize=(10, 6))

    plt.semilogy(err_si, label="Проста ітерація")
    plt.semilogy(err_j, label="Якобі")
    plt.semilogy(err_gs, label="Зейдель")

    plt.xlabel("Ітерації")
    plt.ylabel("Похибка (log scale)")
    plt.title("Порівняння збіжності методів")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()