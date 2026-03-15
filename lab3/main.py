import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    x = []
    y = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))

    return np.array(x), np.array(y)

def form_matrix(x, m):

    n = len(x)
    A = np.zeros((m+1, m+1))

    for i in range(m+1):
        for j in range(m+1):
            s = 0
            for k in range(n):
                s += x[k]**(i+j)
            A[i][j] = s

    return A

def form_vector(x, y, m):

    n = len(x)
    b = np.zeros(m+1)

    for i in range(m+1):
        s = 0
        for k in range(n):
            s += y[k] * x[k]**i
        b[i] = s

    return b

def gauss_solve(A, b):

    n = len(b)

    for k in range(n):

        max_row = k
        for i in range(k+1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i

        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k+1, n):

            factor = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= factor * A[k][j]

            b[i] -= factor * b[k]

    x = np.zeros(n)

    for i in range(n-1, -1, -1):

        s = 0
        for j in range(i+1, n):
            s += A[i][j] * x[j]

        x[i] = (b[i] - s) / A[i][i]

    return x

def polynomial(x, coef):

    y = np.zeros_like(x, dtype=float)

    for i in range(len(coef)):
        y += coef[i] * x**i

    return y

def variance(y_true, y_approx):
    return np.mean((y_true - y_approx)**2)

x = np.array([
1,2,3,4,5,6,7,8,9,10,11,12,
13,14,15,16,17,18,19,20,21,22,23,24
])

y = np.array([
-2,0,5,10,15,20,23,22,17,10,5,0,
-10,3,7,13,19,20,22,21,18,15,10,3
])

max_degree = 4

variances = []
coefs = []

for m in range(1, max_degree+1):

    A = form_matrix(x, m)
    b = form_vector(x, y, m)

    coef = gauss_solve(A.copy(), b.copy())

    y_approx = polynomial(x, coef)

    var = variance(y, y_approx)

    variances.append(var)
    coefs.append(coef)

    print(f"Степінь {m} -> дисперсія = {var:.4f}")

optimal_m = np.argmin(variances) + 1

print("\nОптимальний степінь полінома:", optimal_m)

coef = coefs[optimal_m-1]

y_approx = polynomial(x, coef)

x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef)

print("\nПрогноз температур:")
for i in range(3):
    print(f"Місяць {x_future[i]} -> {y_future[i]:.2f} °C")

error = y - y_approx

plt.figure()

plt.scatter(x, y, label="Фактичні дані")
plt.plot(x, y_approx, label="Апроксимація")

plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.legend()
plt.title("Апроксимація температур")

plt.show()

plt.figure()

plt.plot(x, error, marker='o')

plt.xlabel("Місяць")
plt.ylabel("Похибка")

plt.title("Похибка апроксимації")

plt.show()

plt.figure()

degrees = range(1, max_degree+1)

plt.plot(degrees, variances, marker='o')

plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")

plt.title("Залежність дисперсії від степеня")

plt.show()