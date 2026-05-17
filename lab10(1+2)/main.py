
# ЧАСТИНА 1:
# Метод прогнозу-корекції Адамса 2-го порядку

# ЧАСТИНА 2:
# Метод Рунге-Кутта 4-го порядку

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return y - x ** 2 + 1

def exact_solution(x):
    return (x + 1) ** 2 - 0.5 * np.exp(x)

x0 = 0
y0 = 0.5

a = 0
b = 2

h = 0.1

eps = 1e-5

def runge_kutta_4(f, x0, y0, h, b):
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < b - 1e-12:
        if x + h > b:
            h = b - x

        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x + h

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)

def local_error_exact(x_values, y_values):
    exact = exact_solution(x_values)
    return np.abs(exact - y_values)

def runge_error(f, x0, y0, h, b, p=4):
    x_h, y_h = runge_kutta_4(f, x0, y0, h, b)
    x_h2, y_h2 = runge_kutta_4(f, x0, y0, h / 2, b)

    y_h2_compare = y_h2[::2]
    error = np.abs(y_h2_compare - y_h) / (2 ** p - 1)

    return x_h, error

def automatic_step_rk4(f, x0, y0, b, eps):
    x_values = [x0]
    y_values = [y0]
    h_values = [0.1]

    x = x0
    y = y0
    h = 0.1
    h_max = 0.1

    while x < b - 1e-12:
        if x + h > b:
            h = b - x

        _, y_big = runge_kutta_4(f, x, y, h, x + h)
        _, y_small = runge_kutta_4(f, x, y, h / 2, x + h)

        y1 = y_big[-1]
        y2 = y_small[-1]

        error = abs(y2 - y1) / 15

        if error > eps:
            h = h / 2
            continue
        elif error < eps / 32:
            h = h * 2
            if h > h_max:
                h = h_max

        x = x + h
        y = y2

        x_values.append(x)
        y_values.append(y)
        h_values.append(h)

    return np.array(x_values), np.array(y_values), np.array(h_values)

def adams_predictor_corrector(f, x0, y0, h, b):

    x_rk, y_rk = runge_kutta_4(f, x0, y0, h, x0 + h)

    x_values = list(x_rk)
    y_values = list(y_rk)

    error_estimates = [0, 0]
    x = x_values[-1]

    while x < b - 1e-12:
        if x + h > b:
            h = b - x

        n = len(x_values) - 1
        xn, yn = x_values[n], y_values[n]
        xn1, yn1 = x_values[n - 1], y_values[n - 1]

        fn = f(xn, yn)
        fn1 = f(xn1, yn1)

        y_predict = yn + h * (3 * fn - fn1) / 2

        y_j = y_predict
        for _ in range(2):
            y_j = yn + h * (f(xn + h, y_j) + fn) / 2

        y_correct = y_j

        error_estimate = abs(y_correct - y_predict) / 6
        error_estimates.append(error_estimate)

        x_new = xn + h
        x_values.append(x_new)
        y_values.append(y_correct)
        x = x_new

    return np.array(x_values), np.array(y_values), np.array(error_estimates)

def automatic_step_adams(f, x0, y0, b, eps):
    x_list = [x0]
    y_list = [y0]
    h_list = [0.1]

    h = 0.1
    x = x0
    y = y0

    while x < b - 1e-12:

        if x + 2 * h > b:
            h = (b - x) / 2
            if h < 1e-12:
                break

        _, y_rk = runge_kutta_4(f, x, y, h, x + h)
        x_1, y_1 = x + h, y_rk[-1]

        f1 = f(x_1, y_1)
        f0 = f(x, y)
        y_pred = y_1 + h * (3 * f1 - f0) / 2

        x_2 = x_1 + h
        y_j = y_pred
        for _ in range(2):
            y_j = y_1 + h * (f(x_2, y_j) + f1) / 2
        y_cor = y_j

        error = abs(y_cor - y_pred) / 6

        if error > eps:
            h = h / 2
            continue

        x_list.append(x_1)
        y_list.append(y_1)
        h_list.append(h)

        x_list.append(x_2)
        y_list.append(y_cor)
        h_list.append(h)

        x = x_2
        y = y_cor

        if error < eps / 8:
            h = h * 2

    return np.array(x_list), np.array(y_list), np.array(h_list)

def plot_solution(x_num, y_num, title):
    x_exact = np.linspace(a, b, 500)
    y_exact = exact_solution(x_exact)

    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, label='Точний розв’язок')
    plt.plot(x_num, y_num, 'o-', label='Чисельний розв’язок')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error(x, error, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, 'r')
    plt.xlabel('x')
    plt.ylabel('Похибка')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_step(x, h_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, h_values, 'g')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.title(title)
    plt.grid(True)
    plt.show()

print("=" * 60)
print("ЧАСТИНА 1")
print("МЕТОД ПРОГНОЗУ-КОРЕКЦІЇ АДАМСА")
print("=" * 60)

x_adams, y_adams, adams_est_error = adams_predictor_corrector(f, x0, y0, h, b)

print("\nТАБЛИЦЯ ЗНАЧЕНЬ:")
for i in range(len(x_adams)):
    print(f"x = {x_adams[i]:.2f}   y = {y_adams[i]:.8f}")

plot_solution(x_adams, y_adams, "Метод Адамса прогнозу-корекції")

adams_exact_error = local_error_exact(x_adams, y_adams)
plot_error(x_adams, adams_exact_error, "Локальна похибка Адамса (точний розв’язок)")
plot_error(x_adams, adams_est_error, "Оцінка похибки Адамса (Теоретична формула)")

x_auto_adams, y_auto_adams, h_auto_adams = automatic_step_adams(f, x0, y0, b, eps)
plot_solution(x_auto_adams, y_auto_adams, "Адамс з автоматичним вибором кроку")
plot_step(x_auto_adams, h_auto_adams, "Залежність кроку h(x) для Адамса")

print("\n" + "=" * 60)
print("ЧАСТИНА 2")
print("МЕТОД РУНГЕ-КУТТА 4 ПОРЯДКУ")
print("=" * 60)

x_rk, y_rk = runge_kutta_4(f, x0, y0, h, b)

print("\nТАБЛИЦЯ ЗНАЧЕНЬ:")
for i in range(len(x_rk)):
    print(f"x = {x_rk[i]:.2f}   y = {y_rk[i]:.8f}")

plot_solution(x_rk, y_rk, "Метод Рунге-Кутта 4-го порядку")

rk_exact_error = local_error_exact(x_rk, y_rk)
plot_error(x_rk, rk_exact_error, "Локальна похибка RK4 (точний розв'язок)")

x_err, runge_err = runge_error(f, x0, y0, h, b)
plot_error(x_err, runge_err, "Похибка RK4 за методом Рунге")

max_error = np.max(runge_err)
h_opt = h * (eps / max_error) ** (1 / 4)

print("\nОЦІНКА ОПТИМАЛЬНОГО КРОКУ:")
print(f"Поточний крок h = {h}")
print(f"Максимальна похибка за Рунге = {max_error:e}")
print(f"Рекомендований стаціонарний крок h_opt = {h_opt:e}")

x_auto_rk, y_auto_rk, h_auto_rk = automatic_step_rk4(f, x0, y0, b, eps)
plot_solution(x_auto_rk, y_auto_rk, "RK4 з автоматичним вибором кроку")
plot_step(x_auto_rk, h_auto_rk, "Залежність кроку h(x) для RK4")

steps = [0.5, 0.25, 0.1, 0.05, 0.025]
max_errors = []

for step in steps:
    x_tmp, y_tmp = runge_kutta_4(f, x0, y0, step, b)
    err = local_error_exact(x_tmp, y_tmp)
    max_errors.append(np.max(err))

plt.figure(figsize=(10, 6))
plt.plot(steps, max_errors, 'o-')
plt.xlabel('Крок h')
plt.ylabel('Максимальна похибка')
plt.title('Залежність похибки від величини кроку')
plt.grid(True)
plt.show()

print("\nРоботу успішно завершено.")