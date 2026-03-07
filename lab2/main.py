import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Вхідні дані (варіант 5)
# -------------------------

objects = [100, 200, 400, 800, 1600]
fps = [120, 110, 90, 65, 40]


# -------------------------
# Таблиця розділених різниць
# -------------------------

def divided_differences(x, y):

    n = len(y)
    coef = y.copy()

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    return coef


# -------------------------
# Поліном Ньютона
# -------------------------

def newton_polynomial(x_data, coef, x):

    n = len(coef) - 1
    result = coef[n]

    for k in range(1, n + 1):
        result = coef[n - k] + (x - x_data[n - k]) * result

    return result


# -------------------------
# Основна частина
# -------------------------

coef = divided_differences(objects, fps)

# прогноз FPS для 1000 об'єктів
fps_1000 = newton_polynomial(objects, coef, 1000)

print("FPS для 1000 об'єктів =", fps_1000)


# -------------------------
# пошук межі FPS >= 60
# -------------------------

limit = None

for obj in range(100, 1600):

    value = newton_polynomial(objects, coef, obj)

    if value < 60:
        limit = obj
        break

print("FPS падає нижче 60 приблизно при:", limit, "об'єктах")


# -------------------------
# Побудова графіка
# -------------------------

x_vals = np.linspace(100, 1600, 200)
y_vals = [newton_polynomial(objects, coef, x) for x in x_vals]

plt.scatter(objects, fps, label="Експериментальні дані")
plt.plot(x_vals, y_vals, label="Інтерполяційний поліном Ньютона")

plt.xlabel("Objects")
plt.ylabel("FPS")
plt.title("Залежність FPS від кількості об'єктів")

plt.legend()
plt.grid()

plt.show()