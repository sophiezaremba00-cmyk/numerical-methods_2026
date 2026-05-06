import math
import cmath
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3 - 3*x + 1

def df(x):
    return 3*x**2 - 3

def tabulate(a, b, h):
    print("\n--- ТАБУЛЯЦІЯ ---")
    with open("table.txt", "w") as file:
        x = a
        while x <= b:
            y = f(x)
            print(f"x={x:.2f}, f(x)={y:.4f}")
            file.write(f"{x} {y}\n")
            x += h

def plot_function():
    x = np.linspace(-3, 3, 400)
    y = f(x)

    plt.figure()
    plt.axhline(0)
    plt.axvline(0)
    plt.plot(x, y, label='f(x) = x^3 - 3x + 1')
    plt.grid()
    plt.legend()
    plt.title("Графік трансцендентної функції")
    plt.show()

def plot_polynomial(coeffs):
    x = np.linspace(-3, 3, 400)
    y = np.polyval(coeffs, x)

    plt.figure()
    plt.axhline(0)
    plt.axvline(0)
    plt.plot(x, y, label='Polynomial')
    plt.grid()
    plt.legend()
    plt.title("Графік алгебраїчного рівняння")
    plt.show()

def read_coeffs(filename):
    with open(filename, "r") as f:
        return [float(x) for x in f.read().split()]

def horner(a, x):
    res = a[0]
    for coef in a[1:]:
        res = res * x + coef
    return res

def horner_derivative(a, x):
    n = len(a) - 1
    b = [a[0]]
    for i in range(1, n):
        b.append(b[i-1]*x + a[i])
    return horner(b, x)

def newton(x0, eps):
    x = x0
    it = 0
    while True:
        x1 = x - f(x)/df(x)
        it += 1
        if abs(x1 - x) < eps and abs(f(x1)) < eps:
            return x1, it
        x = x1

def simple_iteration(x0, eps):
    x = x0
    it = 0
    h = 1.0 / max(abs(df(x0)), 1)

    while True:
        x1 = x - h * f(x)
        it += 1
        if abs(x1 - x) < eps and abs(f(x1)) < eps:
            return x1, it
        x = x1

def chebyshev(x0, eps):
    x = x0
    it = 0
    while True:
        f1 = f(x)
        f2 = df(x)
        f3 = 6*x

        x1 = x - f1/f2 - (f1**2 * f3)/(2 * f2**3)
        it += 1

        if abs(x1 - x) < eps and abs(f(x1)) < eps:
            return x1, it

        x = x1

def chord(x0, x1, eps):
    it = 0
    while True:
        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        it += 1
        if abs(x2 - x1) < eps and abs(f(x2)) < eps:
            return x2, it
        x0, x1 = x1, x2

def muller(x0, x1, x2, eps):
    it = 0
    while True:
        f0, f1, f2 = f(x0), f(x1), f(x2)

        h1 = x1 - x0
        h2 = x2 - x1

        d1 = (f1 - f0)/h1
        d2 = (f2 - f1)/h2

        a = (d2 - d1)/(h1 + h2)
        b = a*h2 + d2
        c = f2

        D = math.sqrt(abs(b*b - 4*a*c))

        if abs(b + D) > abs(b - D):
            E = b + D
        else:
            E = b - D

        x3 = x2 + (-2*c)/E
        it += 1

        if abs(x3 - x2) < eps and abs(f(x3)) < eps:
            return x3, it

        x0, x1, x2 = x1, x2, x3

def inverse_interp(x0, x1, x2, eps):
    it = 0
    while True:
        f0, f1, f2 = f(x0), f(x1), f(x2)

        x3 = (
            x0*f1*f2/((f0-f1)*(f0-f2)) +
            x1*f0*f2/((f1-f0)*(f1-f2)) +
            x2*f0*f1/((f2-f0)*(f2-f1))
        )

        it += 1

        if abs(x3 - x2) < eps and abs(f(x3)) < eps:
            return x3, it

        x0, x1, x2 = x1, x2, x3

def newton_horner(a, x0, eps):
    x = x0
    it = 0
    while True:
        fx = horner(a, x)
        dfx = horner_derivative(a, x)

        x1 = x - fx/dfx
        it += 1

        if abs(x1 - x) < eps:
            return x1, it

        x = x1

def lina_method(a, eps):
    p, q = 0.0, 0.0
    it = 0

    while True:
        n = len(a) - 1
        b = [0]*(n+1)

        b[n] = a[n]
        b[n-1] = a[n-1] + p*b[n]

        for i in range(n-2, -1, -1):
            b[i] = a[i] + p*b[i+1] + q*b[i+2]

        if abs(b[2]) < 1e-12:
            break

        p_new = b[1] / b[2]
        q_new = b[0] / b[2]

        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            p, q = p_new, q_new
            break

        p, q = p_new, q_new
        it += 1

    D = p**2 - 4*q
    x1 = (-p + cmath.sqrt(D)) / 2
    x2 = (-p - cmath.sqrt(D)) / 2

    return x1, x2, it

if __name__ == "__main__":
    eps = 1e-6

    tabulate(-3, 3, 0.5)

    plot_function()

    starts = [-2, 0.5]

    print("\n--- НЕЛІНІЙНЕ РІВНЯННЯ ---")
    for x0 in starts:
        print(f"\nПочаткове x0 = {x0}")
        print("Newton:", newton(x0, eps))
        print("Simple:", simple_iteration(x0, eps))
        print("Chebyshev:", chebyshev(x0, eps))
        print("Chord:", chord(x0, x0+0.5, eps))
        print("Muller:", muller(x0, x0+0.5, x0+1, eps))
        print("Inverse:", inverse_interp(x0, x0+0.5, x0+1, eps))

    coeffs = read_coeffs("coeffs.txt")

    plot_polynomial(coeffs)

    print("\n--- АЛГЕБРАЇЧНЕ РІВНЯННЯ ---")
    print("Newton+Horner:", newton_horner(coeffs, -1, eps))
    print("Lina:", lina_method(coeffs, eps))