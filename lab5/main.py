import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi

call_count = 0

def f(x):
    global call_count
    call_count += 1
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

a, b = 0, 24

def exact_integral():
    I1 = 50 * (b - a)

    I2 = (-240 / pi) * (np.cos(np.pi * b / 12) - np.cos(np.pi * a / 12))

    I3 = 5 * sqrt(pi / 0.2) * erf(sqrt(0.2) * 12)

    return I1 + I2 + I3


I0 = exact_integral()
print("Точне значення інтегралу I0 =", I0)

def simpson(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N має бути парним!")

    h = (b - a) / N
    s = f(a) + f(b)

    for i in range(1, N):
        x = a + i * h
        if i % 2 == 0:
            s += 2 * f(x)
        else:
            s += 4 * f(x)

    return s * h / 3

N_values = list(range(10, 1001, 10))
errors = []

for N in N_values:
    if N % 2 != 0:
        N += 1
    call_count = 0
    I = simpson(f, a, b, N)
    errors.append(abs(I - I0))

plt.figure(figsize=(10, 6))
plt.plot(N_values, errors)
plt.yscale("log")
plt.title("Залежність похибки від N (метод Сімпсона)")
plt.xlabel("N")
plt.ylabel("Похибка")
plt.grid()
plt.show()

N_test = 100
call_count = 0
I_test = simpson(f, a, b, N_test)
err_test = abs(I_test - I0)

print(f"\nПохибка при N = {N_test}: {err_test}")

eps_target = 1e-12
N_opt = None

for N in range(10, 10000, 2):
    call_count = 0
    I = simpson(f, a, b, N)
    err = abs(I - I0)
    if err < eps_target:
        N_opt = N
        eps_opt = err
        break

print("\nN_opt =", N_opt)
print("Досягнута точність =", eps_opt)

def runge_romberg(I_h, I_2h, p=4):
    return I_2h + (I_2h - I_h) / (2 ** p - 1)


N = 100
I_h = simpson(f, a, b, N)
I_2h = simpson(f, a, b, 2 * N)

I_rr = runge_romberg(I_h, I_2h)

print("\nМетод Рунге–Ромберга:")
print("I_h =", I_h)
print("I_2h =", I_2h)
print("Уточнене значення =", I_rr)
print("Похибка =", abs(I_rr - I0))

def aitken_order(I_h, I_2h, I_4h):
    return np.log2(abs((I_h - I_2h) / (I_2h - I_4h)))


def aitken_refinement(I_h, I_2h, p):
    return I_2h + (I_2h - I_h) / (2**p - 1)


N = 50
I_h = simpson(f, a, b, N)
I_2h = simpson(f, a, b, 2 * N)
I_4h = simpson(f, a, b, 4 * N)

p_est = aitken_order(I_h, I_2h, I_4h)
I_aitken = aitken_refinement(I_h, I_2h, p_est)

print("\nМетод Ейткена:")
print("Оцінка порядку p =", p_est)
print("Уточнене значення =", I_aitken)
print("Похибка =", abs(I_aitken - I0))

def adaptive_simpson(f, a, b, eps, max_depth=10):
    def simpson_local(f, a, b):
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

    def recurse(f, a, b, eps, whole, depth):
        c = (a + b) / 2
        left = simpson_local(f, a, c)
        right = simpson_local(f, c, b)

        if depth <= 0 or abs(left + right - whole) < 15 * eps:
            return left + right + (left + right - whole) / 15

        return (recurse(f, a, c, eps / 2, left, depth - 1) +
                recurse(f, c, b, eps / 2, right, depth - 1))

    initial = simpson_local(f, a, b)
    return recurse(f, a, b, eps, initial, max_depth)


eps_values = [1e-3, 1e-6, 1e-9, 1e-12]
adaptive_errors = []

print("\nЗалежність точності від кількості викликів функції:")

for eps in eps_values:
    call_count = 0
    I_adapt = adaptive_simpson(f, a, b, eps)
    err = abs(I_adapt - I0)
    adaptive_errors.append(err)

    print(f"eps = {eps:1.0e} | похибка = {err:.3e} | викликів f = {call_count}")

plt.figure(figsize=(10, 6))
plt.plot(eps_values, adaptive_errors, marker='o')
plt.xscale("log")
plt.yscale("log")
plt.title("Похибка адаптивного методу")
plt.xlabel("eps")
plt.ylabel("Похибка")
plt.grid()
plt.show()

call_count = 0
I_normal = simpson(f, a, b, 1000)
calls_normal = call_count

call_count = 0
I_adapt = adaptive_simpson(f, a, b, 1e-6)
calls_adapt = call_count

print("\nПорівняння:")
print("Сімпсон: викликів =", calls_normal)
print("Адаптивний: викликів =", calls_adapt)

x = np.linspace(a, b, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Графік функції")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()