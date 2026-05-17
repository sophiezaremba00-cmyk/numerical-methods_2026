import math
import matplotlib.pyplot as plt
import numpy as np

# ВИБІР РЕЖИМУ РОБОТИ

# 1 - тестування на функції Розенброка
# 2 - розв'язування системи нелінійних рівнянь

MODE = 2

# ВИБІР ПОЧАТКОВОГО НАБЛИЖЕННЯ

# True  -> перший корінь
# False -> другий корінь

FIRST_ROOT = True

def rosenbrock(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

def system_function(x, y):
    return (x ** 2 + y ** 2 - 4) ** 2 + (x - y - 1) ** 2

def F(x, y):

    if MODE == 1:
        return rosenbrock(x, y)

    return system_function(x, y)

def exploratory_search(x, y, h):

    current_value = F(x, y)

    new_x = x + h

    if F(new_x, y) < current_value:
        x = new_x
    else:
        new_x = x - h

        if F(new_x, y) < current_value:
            x = new_x

    current_value = F(x, y)

    new_y = y + h

    if F(x, new_y) < current_value:
        y = new_y
    else:
        new_y = y - h

        if F(x, new_y) < current_value:
            y = new_y

    return x, y

def hooke_jeeves(start_x, start_y):

    x = start_x
    y = start_y

    h = 0.5

    eps_x = 0.001
    eps_f = 0.000001

    alpha = 2.0

    trajectory = []

    steps = 0

    while h > eps_x:

        old_x = x
        old_y = y

        old_f = F(old_x, old_y)

        x, y = exploratory_search(x, y, h)

        new_f = F(x, y)

        if new_f < old_f:

            pattern_x = x + (x - old_x)
            pattern_y = y + (y - old_y)

            pattern_x, pattern_y = exploratory_search(
                pattern_x,
                pattern_y,
                h
            )

            if F(pattern_x, pattern_y) < F(x, y):
                x = pattern_x
                y = pattern_y

            trajectory.append((x, y, F(x, y)))

            steps += 1

            if abs(F(x, y) - old_f) < eps_f:
                break

        else:

            h = h / alpha

    return x, y, F(x, y), trajectory, steps

if FIRST_ROOT:
    start_x = 1.0
    start_y = 1.0
else:
    start_x = -1.0
    start_y = -1.0


x_min, y_min, f_min, trajectory, steps = hooke_jeeves(
    start_x,
    start_y
)


print("\n========================================")
print("МЕТОД ХУКА-ДЖИВСА")
print("========================================")

if MODE == 1:
    print("ТЕСТУВАННЯ НА ФУНКЦІЇ РОЗЕНБРОКА")
else:
    print("РОЗВ'ЯЗУВАННЯ СИСТЕМИ РІВНЯНЬ")

print("----------------------------------------")

print(f"x = {x_min:.6f}")
print(f"y = {y_min:.6f}")
print(f"F(x, y) = {f_min:.10f}")

print(f"Кількість кроків = {steps}")

print("========================================")


with open("trajectory.txt", "w", encoding="utf-8") as file:

    file.write("x\t y\t F(x,y)\n")

    for point in trajectory:

        file.write(
            f"{point[0]:.6f}\t"
            f"{point[1]:.6f}\t"
            f"{point[2]:.10f}\n"
        )

print("\nТраєкторію записано у файл trajectory.txt")

if MODE == 2:

    x_line = np.linspace(-3, 3, 500)
    y_line = x_line - 1

    x_circle = np.linspace(-2, 2, 500)
    y_circle_top = np.sqrt(4 - x_circle ** 2)
    y_circle_bottom = -np.sqrt(4 - x_circle ** 2)

    plt.figure(figsize=(8, 8))

    plt.plot(
        x_circle,
        y_circle_top,
        color='tab:blue',
        label='x² + y² = 4'
    )
    plt.plot(x_circle, y_circle_bottom, color='tab:blue')

    plt.plot(
        x_line,
        y_line,
        color='tab:green',
        label='x - y - 1 = 0'
    )

    plt.scatter(
        x_min,
        y_min,
        s=100,
        label='Розв’язок системи'
    )

    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    plt.plot(
        traj_x,
        traj_y,
        marker='o',
        linestyle='--',
        color='tab:red',
        label='Траєкторія спуску'
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Метод Хука-Дживса")

    plt.grid(True)

    plt.axis('equal')

    plt.legend()

    plt.show()

if MODE == 1:

    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)

    X, Y = np.meshgrid(x, y)

    Z = rosenbrock(X, Y)

    plt.figure(figsize=(8, 6))

    contour = plt.contour(
        X,
        Y,
        Z,
        levels=50
    )

    plt.clabel(contour)

    plt.scatter(
        x_min,
        y_min,
        s=100,
        label='Мінімум'
    )

    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    plt.plot(
        traj_x,
        traj_y,
        marker='o',
        linestyle='--',
        label='Траєкторія спуску'
    )

    plt.title("Функція Розенброка")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)

    plt.legend()

    plt.show()