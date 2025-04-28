x_i = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
y_i = [0, 0.952, 1.849, 2.468, 2.537, 2.138, 1.611, 1.166, 0.842, 0.617, 0.461]

n = len(x_i)

def phi(x):
    return 1.585 - 0.127 * x


# Вычисляем значения для таблицы
phi_values = [phi(x) for x in x_i]
diff_squared = [(phi(x) - y)**2 for x, y in zip(x_i, y_i)]
sum_diff_squared = sum(diff_squared)
sigma = (sum_diff_squared / n) ** 0.5

# Выводим таблицу
print("| i    | " + " | ".join(f"{i+1:<5}" for i in range(n)) + " |")
print("|------|" + "|".join(["-----"]*n) + "|")
print("| xi   | " + " | ".join(f"{x:<5.3f}" for x in x_i) + " |")
print("| yi   | " + " | ".join(f"{y:<5.3f}" for y in y_i) + " |")
print("| φ(xi)| " + " | ".join(f"{p:<5.3f}" for p in phi_values) + " |")
print("| (φ(xi)-yi)² | " + " | ".join(f"{d:<5.3f}" for d in diff_squared) + " |")

# Вычисляем суммы
sx = sum(x_i)
sxx = sum(x**2 for x in x_i)
sy = sum(y_i)
sxy = sum(x * y for x, y in zip(x_i, y_i))
print(f"\nСуммы: sx = {sx}, sxx = {sxx}, sy = {sy}, sxy = {sxy:.3f}")
print(f"Сумма квадратов разностей: ∑(φ(xi)-yi)² = {sum_diff_squared:.5f}")
print(f"Стандартное отклонение: σ = √({sum_diff_squared:.5f}/{n}) = {sigma:.5f}")