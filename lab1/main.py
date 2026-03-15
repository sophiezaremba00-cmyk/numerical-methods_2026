import numpy as np
import matplotlib.pyplot as plt

coords = [
    (48.164214, 24.536044),
    (48.164983, 24.534836),
    (48.165605, 24.534068),
    (48.166228, 24.532915),
    (48.166777, 24.531927),
    (48.167326, 24.530884),
    (48.167011, 24.530061),
    (48.166053, 24.528039),
    (48.166655, 24.526064),
    (48.166497, 24.523574),
    (48.166128, 24.520214),
    (48.165416, 24.517170),
    (48.164546, 24.514640),
    (48.163412, 24.512980),
    (48.162331, 24.511715),
    (48.162015, 24.509462),
    (48.162147, 24.506932),
    (48.161751, 24.504244),
    (48.161197, 24.501793),
    (48.160580, 24.500537),
    (48.160250, 24.500106)
]

elevations = np.array([
    1250, 1270, 1290, 1320, 1350,
    1380, 1400, 1420, 1450, 1470,
    1500, 1520, 1550, 1580, 1610,
    1640, 1680, 1720, 1760, 1800,
    1820
])

n = len(elevations)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)

def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    alpha = np.zeros(n+1)
    for i in range(1, n):
        alpha[i] = (3/h[i])*(y[i+1]-y[i]) - (3/h[i-1])*(y[i]-y[i-1])

    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)

    for i in range(1, n):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    c = np.zeros(n+1)
    b = np.zeros(n)
    d = np.zeros(n)
    a = y[:-1]

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])

    return a, b, c[:-1], d

a, b, c, d = cubic_spline(distances, elevations)

def spline_value(x_val, x, a, b, c, d):
    for i in range(len(a)):
        if x[i] <= x_val <= x[i+1]:
            dx = x_val - x[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

xx = np.linspace(distances[0], distances[-1], 1000)
yy = np.zeros_like(xx)

for i in range(len(a)):
    mask = (xx >= distances[i]) & (xx <= distances[i+1])
    dx = xx[mask] - distances[i]
    yy[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    spline_points = np.array([spline_value(xi, distances, a, b, c, d) for xi in distances])
    errors = elevations - spline_points

total_length = distances[-1]
total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))

print("Загальна довжина маршруту (м):", round(total_length,2))
print("Сумарний набір висоти (м):", round(total_ascent,2))
print("Сумарний спуск (м):", round(total_descent,2))

gradient = np.gradient(yy, xx) * 100

print("Максимальний підйом (%):", round(np.max(gradient),2))
print("Максимальний спуск (%):", round(np.min(gradient),2))
print("Середній градієнт (%):", round(np.mean(np.abs(gradient)),2))

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("Механічна робота (кДж):", round(energy/1000,2))
print("Енергія (ккал):", round(energy/4184,2))

plt.figure(figsize=(10,6))
plt.plot(distances, elevations, 'o', label="Дискретні точки")
plt.plot(xx, yy, label="Кубічний сплайн")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль маршруту Заросляк – Говерла")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(xx, gradient)
plt.xlabel("Відстань (м)")
plt.ylabel("Градієнт (%)")
plt.title("Градієнт маршруту")
plt.grid()
plt.show()

def spline_curve(x_nodes, y_nodes, x_dense):
    a,b,c,d = cubic_spline(x_nodes, y_nodes)
    y_dense = np.zeros_like(x_dense)

    for i in range(len(a)):
        mask = (x_dense >= x_nodes[i]) & (x_dense <= x_nodes[i+1])
        dx = x_dense[mask] - x_nodes[i]
        y_dense[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return y_dense

xx = np.linspace(distances[0], distances[-1], 2000)

ref_x = distances[:20]
ref_y = elevations[:20]

yy_ref = spline_curve(ref_x, ref_y, xx)

idx10 = np.linspace(0,19,10,dtype=int)
x10 = distances[idx10]
y10 = elevations[idx10]

yy10 = spline_curve(x10,y10,xx)

idx15 = np.linspace(0,19,15,dtype=int)
x15 = distances[idx15]
y15 = elevations[idx15]

yy15 = spline_curve(x15,y15,xx)

error10 = np.abs(yy10 - yy_ref)
error15 = np.abs(yy15 - yy_ref)

plt.figure(figsize=(10,6))

plt.plot(xx, error10, color='blue', label="Похибка (10 вузлів)")
plt.plot(xx, error15, color='green', label="Похибка (15 вузлів)")

plt.fill_between(xx, error10, alpha=0.2, color='red')
plt.fill_between(xx, error15, alpha=0.2, color='green')

plt.xlabel("Відстань (м)")
plt.ylabel("Абсолютна похибка (м)")
plt.title("Графік похибки інтерполяції (відносно 20 вузлів)")
plt.legend()
plt.grid()

plt.show()