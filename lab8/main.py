import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 3*x + 1


def df(x):
    return 3*x**2 - 3

def tabulate(a, b, h):

    print(
        f"ПУНКТ 1: Табуляція F(x) "
        f"на [{a}, {b}], h={h}"
    )

    print("=" * 65)

    with open("tabulation.txt", "w") as file:

        file.write(f"{'x':>12} {'F(x)':>18}\n")
        file.write("-" * 30 + "\n")

        print(
            "Таблицю збережено у файл: "
            "tabulation.txt\n"
        )

        print(f"{'x':>12} {'F(x)':>18}")
        print("-" * 30)

        x = a

        roots_intervals = []

        prev_x = x
        prev_y = f(prev_x)

        while x <= b + 1e-12:

            y = f(x)

            print(f"{x:12.4f} {y:18.8f}")

            file.write(f"{x:12.4f} {y:18.8f}\n")

            if prev_y * y < 0:

                x0 = (prev_x + x) / 2

                trend = (
                    "зростаюча"
                    if y > prev_y
                    else "спадна"
                )

                roots_intervals.append(
                    (prev_x, x, x0, trend)
                )

            prev_x = x
            prev_y = y

            x += h

    print("\nТочки перетину з віссю x:")

    for interval in roots_intervals:

        a1, b1, x0, trend = interval

        print(
            f"  [{a1:.2f}, {b1:.2f}]  "
            f"({trend}),  "
            f"x0 ≈ {x0:.6f}"
        )

    print(
        f"\nЗнайдено "
        f"{len(roots_intervals)} "
        f"початкових наближень."
    )

    return roots_intervals

def plot_function():

    x = np.linspace(-3, 3, 500)
    y = f(x)

    plt.figure(figsize=(10, 6))

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.plot(x, y, label='F(x)=x^3-3x+1')

    plt.grid()

    plt.legend()

    plt.title("Графік F(x)")

    plt.savefig("Fx_roots_plot.png")

    print(
        "\nГрафік F(x) "
        "збережено: Fx_roots_plot.png"
    )

    plt.show()

def plot_polynomial(coeffs):

    x = np.linspace(-5, 5, 500)

    y = np.polyval(coeffs, x)

    plt.figure(figsize=(10, 6))

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.plot(x, y, label='Polynomial')

    plt.grid()

    plt.legend()

    plt.title("Графік полінома")

    plt.savefig("poly3_plot.png")

    print(
        "Графік збережено: "
        "poly3_plot.png"
    )

    plt.show()

def read_coeffs(filename):

    with open(filename, "r") as file:

        return [float(x) for x in file.read().split()]


def horner(a, x):

    result = a[0]

    for coef in a[1:]:

        result = result * x + coef

    return result


def horner_derivative(a, x):

    n = len(a) - 1

    b = [a[0]]

    for i in range(1, n):

        b.append(
            b[i - 1] * x + a[i]
        )

    return horner(b, x)


def newton_complex(a, x0, eps):

    x = complex(x0)

    it = 0

    while True:

        fx = horner(a, x)

        dfx = horner_derivative(a, x)

        if abs(dfx) < 1e-14:
            return x, it

        x1 = x - fx / dfx

        it += 1

        if abs(x1 - x) < eps:
            return x1, it

        x = x1


def chord_complex(a, x0, x1, eps):

    x0 = complex(x0)
    x1 = complex(x1)

    it = 0

    while True:

        f0 = horner(a, x0)
        f1 = horner(a, x1)

        if abs(f1 - f0) < 1e-14:
            return x1, it

        x2 = (
            x1
            - f1 * (x1 - x0)
            / (f1 - f0)
        )

        it += 1

        if abs(x2 - x1) < eps:
            return x2, it

        x0, x1 = x1, x2

def muller_complex(a, x0, x1, x2, eps):

    x0 = complex(x0)
    x1 = complex(x1)
    x2 = complex(x2)

    it = 0

    while True:

        f0 = horner(a, x0)
        f1 = horner(a, x1)
        f2 = horner(a, x2)

        h1 = x1 - x0
        h2 = x2 - x1

        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2

        A = (d2 - d1) / (h2 + h1)
        B = A * h2 + d2
        C = f2

        D = cmath.sqrt(B * B - 4 * A * C)

        if abs(B + D) > abs(B - D):
            E = B + D
        else:
            E = B - D

        if abs(E) < 1e-14:
            return x2, it

        x3 = x2 + (-2 * C) / E

        it += 1

        if abs(x3 - x2) < eps:
            return x3, it

        x0, x1, x2 = x1, x2, x3


def inverse_interp_complex(
    a,
    x0,
    x1,
    x2,
    eps
):

    x0 = complex(x0)
    x1 = complex(x1)
    x2 = complex(x2)

    it = 0
    max_iter = 100

    while it < max_iter:

        f0 = horner(a, x0)
        f1 = horner(a, x1)
        f2 = horner(a, x2)

        d01 = (f0 - f1)
        d02 = (f0 - f2)
        d12 = (f1 - f2)

        if abs(d01) < 1e-14:
            d01 += 1e-14

        if abs(d02) < 1e-14:
            d02 += 1e-14

        if abs(d12) < 1e-14:
            d12 += 1e-14

        try:

            x3 = (
                x0 * f1 * f2 / (d01 * d02)
                + x1 * f0 * f2 / ((-d01) * d12)
                + x2 * f0 * f1 / ((-d02) * (-d12))
            )

        except ZeroDivisionError:

            return x2, it

        it += 1

        if abs(x3 - x2) < eps:
            return x3, it

        x0, x1, x2 = x1, x2, x3

    return x2, it

def bairstow(coeffs, r=1, s=1, eps=1e-10):

    a = coeffs[:]

    n = len(a) - 1

    roots = []

    total_iterations = 0

    while n >= 3:

        while True:

            b = [0] * (n + 1)
            c = [0] * (n + 1)

            b[n] = a[n]

            b[n - 1] = (
                a[n - 1]
                + r * b[n]
            )

            for i in range(n - 2, -1, -1):

                b[i] = (
                    a[i]
                    + r * b[i + 1]
                    + s * b[i + 2]
                )

            c[n] = b[n]

            c[n - 1] = (
                b[n - 1]
                + r * c[n]
            )

            for i in range(n - 2, 0, -1):

                c[i] = (
                    b[i]
                    + r * c[i + 1]
                    + s * c[i + 2]
                )

            det = (
                c[2] * c[2]
                - c[3] * c[1]
            )

            if abs(det) < 1e-14:

                r += 1
                s += 1

                continue

            dr = (
                -b[1] * c[2]
                + b[0] * c[3]
            ) / det

            ds = (
                -b[0] * c[2]
                + b[1] * c[1]
            ) / det

            r += dr
            s += ds

            total_iterations += 1

            if (
                abs(dr) < eps
                and abs(ds) < eps
            ):
                break

        D = r * r + 4 * s

        if D >= 0:

            x1 = (
                r + math.sqrt(D)
            ) / 2

            x2 = (
                r - math.sqrt(D)
            ) / 2

        else:

            x1 = complex(
                r / 2,
                math.sqrt(-D) / 2
            )

            x2 = complex(
                r / 2,
                -math.sqrt(-D) / 2
            )

        roots.append(x1)
        roots.append(x2)

        a = b[2:]

        n -= 2

    if n == 1:

        roots.append(-a[0] / a[1])

    elif n == 2:

        A = a[2]
        B = a[1]
        C = a[0]

        D = B * B - 4 * A * C

        if D >= 0:

            roots.append(
                (-B + math.sqrt(D))
                / (2 * A)
            )

            roots.append(
                (-B - math.sqrt(D))
                / (2 * A)
            )

        else:

            roots.append(
                complex(
                    -B / (2 * A),
                    math.sqrt(-D)
                    / (2 * A)
                )
            )

            roots.append(
                complex(
                    -B / (2 * A),
                    -math.sqrt(-D)
                    / (2 * A)
                )
            )

    return roots, total_iterations

def print_root(name, root, iterations):

    if isinstance(root, complex):

        if abs(root.imag) < 1e-10:

            print(
                f"  {name:<20}: "
                f"x = {root.real:.15f}, "
                f"ітерацій = {iterations}"
            )

        else:

            print(
                f"  {name:<20}: "
                f"x = "
                f"{root.real:.10f} "
                f"{root.imag:+.10f}i, "
                f"ітерацій = {iterations}"
            )

    else:

        print(
            f"  {name:<20}: "
            f"x = {root:.15f}, "
            f"ітерацій = {iterations}"
        )


if __name__ == "__main__":

    eps = 1e-10

    intervals = tabulate(-3.0, 3.0, 0.1)

    plot_function()

    print("\n")

    print("=" * 65)

    print(
        "ПУНКТИ 2–4: "
        "Знаходження коренів усіма методами"
    )

    print(f"Точність: eps = {eps}")

    print("=" * 65)

    coeffs_f = [1, 0, -3, 1]

    starts = [-2, 0.5, 2]

    for idx, x0 in enumerate(starts, start=1):

        print(
            f"\n--- Корінь #{idx}, "
            f"початкове наближення x0 = {x0} ---"
        )

        r, it = newton_complex(
            coeffs_f,
            x0,
            eps
        )

        print_root(
            "Метод Ньютона",
            r,
            it
        )

        r, it = chord_complex(
            coeffs_f,
            x0,
            x0 + 0.5,
            eps
        )

        print_root(
            "Метод хорд",
            r,
            it
        )

        r, it = muller_complex(
            coeffs_f,
            x0,
            x0 + 0.5,
            x0 + 1,
            eps
        )

        print_root(
            "Метод Мюллера",
            r,
            it
        )

        r, it = inverse_interp_complex(
            coeffs_f,
            complex(1.5, 0.5),  # замість complex(0, 1)
            complex(2.5, 1.5),  # замість complex(1, 2)
            complex(2.2, 2.2),  # замість complex(2, 1)
            eps
        )

        print_root(
            "Зворотна інтерп.",
            r,
            it
        )

    print("\n")

    print("=" * 65)

    print("ПУНКТ 5: Алгебраїчне рівняння")

    print("=" * 65)

    coeffs = read_coeffs("coeffs.txt")

    print(
        f"Зчитані коефіцієнти: {coeffs}"
    )

    print("\nРівняння:")

    print(
        "p(x) = x^3 - 4x^2 + 9x - 10 = 0"
    )

    print("\nТеоретичні корені:")

    print(
        "x = 2,  x = 1+2i,  x = 1-2i"
    )

    plot_polynomial(coeffs)

    print("\nМетод Ньютона:")

    r, it = newton_complex(
        coeffs,
        complex(1, 1),
        eps
    )

    print_root(
        "Newton Complex",
        r,
        it
    )

    print("\nМетод хорд:")

    r, it = chord_complex(
        coeffs,
        complex(0, 1),
        complex(1, 2),
        eps
    )

    print_root(
        "Chord Complex",
        r,
        it
    )

    print("\nМетод Мюллера:")

    r, it = muller_complex(
        coeffs,
        complex(0, 0),
        complex(1, 1),
        complex(2, 2),
        eps
    )

    print_root(
        "Muller Complex",
        r,
        it
    )

    print("\nЗворотна інтерполяція:")

    r, it = inverse_interp_complex(
        coeffs,
        complex(0, 1),
        complex(1, 2),
        complex(2, 1),
        eps
    )

    print_root(
        "Inverse Interpolation",
        r,
        it
    )

    print("\n")

    print("=" * 65)

    print(
        "ПУНКТ 9: "
        "Комплексні корені методом Ліна (Берстоу)"
    )

    print("=" * 65)

    roots, total_it = bairstow(
        coeffs[::-1],
        r=1,
        s=-1,
        eps=eps
    )

    print(
        f"Метод Ліна (Берстоу), "
        f"всього ітерацій = {total_it}:"
    )

    for i, root in enumerate(roots, start=1):

        if isinstance(root, complex):

            if abs(root.imag) < 1e-10:

                print(
                    f"  Корінь {i}: "
                    f"x = {root.real:.12f}"
                )

            else:

                print(
                    f"  Корінь {i}: "
                    f"x = "
                    f"{root.real:.8f} "
                    f"{root.imag:+.8f}i"
                )

        else:

            print(
                f"  Корінь {i}: "
                f"x = {root:.12f}"
            )

    print("\nПеревірка через numpy.roots:")

    np_roots = np.roots(coeffs)

    for i, root in enumerate(np_roots, start=1):

        if abs(root.imag) < 1e-10:

            print(
                f"  x{i} = "
                f"{root.real:.12f}"
            )

        else:

            print(
                f"  x{i} = "
                f"{root.real:.8f} "
                f"{root.imag:+.8f}i"
            )


    print("\n")

    print("=" * 65)

    print("ПІДСУМОК")

    print("=" * 65)

    print("Програма успішно виконала:")

    print("  • Табуляцію функції")
    print("  • Побудову графіків")
    print("  • Пошук дійсних коренів")
    print("  • Пошук комплексних коренів")
    print("  • Роботу зі схемою Горнера")
    print("  • Метод Ньютона")
    print("  • Метод хорд")
    print("  • Метод Мюллера")
    print("  • Зворотну інтерполяцію")
    print("  • Метод Ліна (Берстоу)")

    print("\nСтворені файли:")

    print(
        "  tabulation.txt      "
        "— таблиця значень"
    )

    print(
        "  poly3_plot.png      "
        "— графік полінома"
    )

    print(
        "  Fx_roots_plot.png   "
        "— графік функції"
    )

    print("\nГотово!")