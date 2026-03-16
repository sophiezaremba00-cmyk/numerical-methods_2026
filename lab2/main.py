import numpy as np
import matplotlib.pyplot as plt

objects = [100, 200, 400, 800, 1600]
fps = [120, 110, 90, 65, 40]

def divided_differences(x, y):

    n = len(y)
    coef = y.copy()

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    return coef


def newton_polynomial(x_data, coef, x):

    n = len(coef) - 1
    result = coef[n]

    for k in range(1, n + 1):
        result = coef[n - k] + (x - x_data[n - k]) * result

    return result

coef = divided_differences(objects, fps)

fps_1000 = newton_polynomial(objects, coef, 1000)

print("FPS для 1000 об'єктів =", fps_1000)

limit = None

for obj in range(100, 1600):

    value = newton_polynomial(objects, coef, obj)

    if value < 60:
        limit = obj
        break

print("FPS падає нижче 60 приблизно при:", limit, "об'єктах")

x_vals = np.linspace(100, 1600, 200)
y_vals = [newton_polynomial(objects, coef, x) for x in x_vals]

plt.figure()

plt.scatter(objects, fps, label="Експериментальні дані")
plt.plot(x_vals, y_vals, label="Інтерполяційний поліном Ньютона")

plt.xlabel("Objects")
plt.ylabel("FPS")
plt.title("Залежність FPS від кількості об'єктів")

plt.legend()
plt.grid()

plt.show()

x_dense = np.linspace(100, 1600, 400)

plt.figure()

steps = [2, 3, 5]

for step in steps:

    x_nodes = objects[:step]
    y_nodes = fps[:step]

    coef = divided_differences(x_nodes, y_nodes)

    y_interp = [newton_polynomial(x_nodes, coef, x) for x in x_dense]

    plt.plot(x_dense, y_interp, label=f"{step} вузли")

plt.scatter(objects, fps, color="black", label="Дані")

plt.title("Вплив кількості вузлів")
plt.xlabel("Objects")
plt.ylabel("FPS")
plt.legend()
plt.grid()

plt.show()


true_values = [newton_polynomial(objects, divided_differences(objects, fps), x) for x in x_dense]

errors = []

nodes_counts = [2, 3, 4, 5]

for n in nodes_counts:

    x_nodes = objects[:n]
    y_nodes = fps[:n]

    coef = divided_differences(x_nodes, y_nodes)

    approx = [newton_polynomial(x_nodes, coef, x) for x in x_dense]

    error = np.mean(np.abs(np.array(true_values) - np.array(approx)))

    errors.append(error)

plt.figure()

plt.plot(nodes_counts, errors, marker="o")

plt.xlabel("Кількість вузлів")
plt.ylabel("Середня похибка")
plt.title("Залежність похибки від кількості вузлів")

plt.grid()

plt.show()

def runge_function(x):
    return 1 / (1 + 25 * x**2)

x = np.linspace(-1, 1, 400)

plt.figure()

for n in [5, 10, 15]:

    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    coef = divided_differences(list(x_nodes), list(y_nodes))

    y_interp = [newton_polynomial(list(x_nodes), coef, xi) for xi in x]

    plt.plot(x, y_interp, label=f"{n} вузлів")

plt.plot(x, runge_function(x), linewidth=3, label="Справжня функція")

plt.title("Ефект Рунге")
plt.legend()
plt.grid()

plt.show()