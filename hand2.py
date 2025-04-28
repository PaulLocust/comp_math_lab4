import numpy as np

# Исходные данные
xi = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
yi = [0, 0.952, 1.849, 2.468, 2.537, 2.138, 1.611, 1.166, 0.842, 0.617, 0.461]
n = len(xi)

# Вычисление сумм
sx = sum(xi)
sxx = sum(x**2 for x in xi)
sxxx = sum(x**3 for x in xi)
sxxxx = sum(x**4 for x in xi)
sy = sum(yi)
sxy = sum(x*y for x,y in zip(xi,yi))
sxxy = sum(x**2*y for x,y in zip(xi,yi))

# Матрица системы уравнений
A = np.array([
    [n, sx, sxx],
    [sx, sxx, sxxx],
    [sxx, sxxx, sxxxx]
])

# Вектор правой части
B = np.array([sy, sxy, sxxy])

# Решение системы методом Крамера
def cramer(A, B):
    det = np.linalg.det(A)
    det1 = np.linalg.det(np.array([B, A[:,1], A[:,2]]).T)
    det2 = np.linalg.det(np.array([A[:,0], B, A[:,2]]).T)
    det3 = np.linalg.det(np.array([A[:,0], A[:,1], B]).T)
    print(det, det1, det2, det3)
    return det1/det, det2/det, det3/det

a, b, c = cramer(A, B)

# Функция аппроксимации
def phi(x):
    return a + b*x + c*x**2

# Вычисление значений φ(xi) и квадратов разностей
phi_values = [phi(x) for x in xi]
diff_squared = [(phi(x)-y)**2 for x,y in zip(xi,yi)]
sum_diff_squared = sum(diff_squared)
sigma = (sum_diff_squared / n)**0.5

# Вывод результатов
print("Суммы:")
print(f"sx = {sx}, sxx = {sxx}, sxxx = {sxxx}, sxxxx = {sxxxx}")
print(f"sy = {sy}, sxy = {sxy}, sxxy = {sxxy}\n")

print("Система уравнений:")
print(f"{n}a + {sx}b + {sxx}c = {sy}")
print(f"{sx}a + {sxx}b + {sxxx}c = {sxy}")
print(f"{sxx}a + {sxxx}b + {sxxxx}c = {sxxy}\n")

print(f"Коэффициенты: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")
print(f"Аппроксимирующая функция: φ(x) = {a:.3f} + {b:.3f}x + {c:.3f}x²\n")

# Вывод таблицы
print("| i  |   xi  |   yi   | φ(xi) | (φ(xi)-yi)² |")
print("|----|-------|--------|-------|-------------|")
for i in range(n):
    print(f"| {i+1:2} | {xi[i]:.1f} | {yi[i]:6.3f} | {phi_values[i]:5.3f} | {diff_squared[i]:7.3f}    |")

print(f"\nСумма квадратов разностей: ∑(φ(xi)-yi)² = {sum_diff_squared:.3f}")
print(f"Стандартное отклонение: σ = √({sum_diff_squared:.3f}/{n}) = {sigma:.5f}")