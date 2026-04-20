import math
import matplotlib.pyplot as plt
import numpy as np

def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def dM_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

t0 = 1
exact = dM_exact(t0)

print("1. Аналітичне значення")
print(f"M'(1) = {exact:.10f}\n")

def derivative_central(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

print("2. Залежність від кроку h")

h_values = [0.5, 0.1, 0.05, 0.01, 0.005]
errors = []
approximations = []

for h in h_values:
    d = derivative_central(t0, h)
    approximations.append(d)
    error = abs(d - exact)
    errors.append(error)
    print(f"h = {h:<8} D(h) = {d:.10f}  похибка = {error:.10e}")

print()

h_opt = 0.01
print("3. Обраний оптимальний крок")
print(f"h_opt = {h_opt}")

d_opt = derivative_central(t0, h_opt)
error_opt = abs(d_opt - exact)
print(f"Досягнута похибка = {error_opt:.10e}\n")

print("4. Два кроки")
h = h_opt
d_h = derivative_central(t0, h)
d_h2 = derivative_central(t0, h / 2)

print(f"D(h)     = {d_h:.10f}")
print(f"D(h/2)   = {d_h2:.10f}\n")

print("5. Похибка при h")
error_h = abs(d_h - exact)
print(f"ε(h) = {error_h:.10e}\n")

print("6. Метод Рунге–Ромберга")

p = 2
D_runge = d_h2 + (d_h2 - d_h) / (2**p - 1)
error_runge = abs(D_runge - exact)

print(f"D_RR = {D_runge:.10f}")
print(f"Похибка RR = {error_runge:.10e}")
print(f"Зменшення похибки ≈ {error_h / error_runge:.2f} рази\n")

print("7. Метод Ейткена")

d_h4 = derivative_central(t0, h / 4)

D_aitken = d_h - ((d_h2 - d_h) ** 2) / (d_h4 - 2 * d_h2 + d_h)
error_aitken = abs(D_aitken - exact)

p_est = math.log(abs((d_h2 - d_h) / (d_h4 - d_h2))) / math.log(2)

print(f"D* (Aitken) = {D_aitken:.10f}")
print(f"Похибка Aitken = {error_aitken:.10e}")
print(f"Оцінка порядку точності p ≈ {p_est:.2f}\n")

print("ВИСНОВОК:")
print("- Видно 2 порядок точності методу")
print("- Метод Рунге–Ромберга зменшує похибку")
print("- Метод Ейткена підтверджує порядок точності ≈ 2\n")

t_vals = np.linspace(0, 10, 200)
M_vals = [M(t) for t in t_vals]

plt.figure()
plt.plot(t_vals, M_vals)
plt.title("Графік функції M(t)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.grid(True)

plt.figure()
plt.loglog(h_values, errors, marker='o')
plt.title("Залежність похибки від кроку h")
plt.xlabel("h (лог шкала)")
plt.ylabel("Похибка (лог шкала)")
plt.grid(True)

plt.figure()
plt.plot(h_values, approximations, marker='o', label="Чисельне значення")
plt.axhline(y=exact, linestyle='--', label="Точне значення")
plt.gca().invert_xaxis()

plt.title("Збіжність чисельної похідної до точної")
plt.xlabel("h")
plt.ylabel("M'(1)")
plt.legend()
plt.grid(True)

plt.show()