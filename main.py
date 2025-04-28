import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import locale
from matrix import solve_sle

# Установим локаль для корректного распознавания разделителей
locale.setlocale(locale.LC_ALL, '')


def parse_number(s):
    """Преобразует строку в число, обрабатывая как точки, так и запятые в качестве разделителей"""
    try:
        # Сначала пробуем стандартное преобразование
        return float(s)
    except ValueError:
        # Если не получилось, заменяем запятую на точку и пробуем снова
        try:
            return float(s.replace(',', '.'))
        except ValueError:
            # Если все равно не получается, возвращаем None
            return None


def mean(values):
    """Вычисляет среднее значение"""
    return sum(values) / len(values) if len(values) > 0 else 0


def correlation_coefficient(x, y):
    """Вычисляет коэффициент корреляции Пирсона"""
    n = len(x)
    if n != len(y):
        return 0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0

    return numerator / (denominator_x * denominator_y)


# Методы аппроксимации
def linear_approximation(x, y):
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx = sum(i ** 2 for i in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    denominator = n * sxx - sx ** 2
    if denominator == 0:
        return None
    b = (n * sxy - sx * sy) / denominator
    a = (sy - b * sx) / n
    return (a, b), lambda t: a + b * t


def quadratic_approximation(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(i ** 2 for i in x)
    sxxx = sum(i ** 3 for i in x)
    sxxxx = sum(i ** 4 for i in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    sxxy = sum((x[i] ** 2) * y[i] for i in range(n))

    # Создаем матрицу коэффициентов и вектор правой части
    A = [
        [n, sx, sxx],
        [sx, sxx, sxxx],
        [sxx, sxxx, sxxxx]
    ]
    B = [sy, sxy, sxxy]

    try:
        a, b, c = solve_sle(A, B, 3)
        return (a, b, c), lambda t: a + b * t + c * t ** 2
    except:
        return None


def cubic_approximation(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(i ** 2 for i in x)
    sxxx = sum(i ** 3 for i in x)
    sxxxx = sum(i ** 4 for i in x)
    sxxxxx = sum(i ** 5 for i in x)
    sxxxxxx = sum(i ** 6 for i in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    sxxy = sum((x[i] ** 2) * y[i] for i in range(n))
    sxxxy = sum((x[i] ** 3) * y[i] for i in range(n))

    A = [
        [n, sx, sxx, sxxx],
        [sx, sxx, sxxx, sxxxx],
        [sxx, sxxx, sxxxx, sxxxxx],
        [sxxx, sxxxx, sxxxxx, sxxxxxx]
    ]
    B = [sy, sxy, sxxy, sxxxy]

    try:
        a, b, c, d = solve_sle(A, B, 4)
        return (a, b, c, d), lambda t: a + b * t + c * t ** 2 + d * t ** 3
    except:
        return None


def exponential_approximation(x, y):
    try:
        x_valid, y_valid = zip(*[(x[i], y[i]) for i in range(len(x)) if y[i] > 0])
        ln_y = [math.log(i) for i in y_valid]
        coeffs, f = linear_approximation(x_valid, ln_y)
        if coeffs is None:
            return None
        a = math.exp(coeffs[0])
        b = coeffs[1]
        return (a, b), lambda t: a * math.exp(b * t)
    except:
        return None


def logarithmic_approximation(x, y):
    try:
        # Исключаем значения x <= 0, так как логарифм не определён для таких значений
        x_valid, y_valid = zip(*[(x[i], y[i]) for i in range(len(x)) if x[i] > 0])

        # Применяем логарифм только к положительным значениям x
        ln_x = [math.log(i) for i in x_valid]

        coeffs, f = linear_approximation(ln_x, y_valid)
        if coeffs is None:
            return None

        a, b = coeffs
        return (a, b), lambda t: a + b * math.log(t) if t > 0 else float('nan')  # Защита от отрицательных значений t
    except Exception as e:
        print(f"Ошибка при аппроксимации логарифмической функцией: {e}")
        return None


def power_approximation(x, y):
    try:
        x_valid, y_valid = zip(*[(x[i], y[i]) for i in range(len(x)) if x[i] > 0 and y[i] > 0])
        ln_x = [math.log(i) for i in x_valid]
        ln_y = [math.log(i) for i in y_valid]
        coeffs, f = linear_approximation(ln_x, ln_y)
        if coeffs is None:
            return None
        a = math.exp(coeffs[0])
        b = coeffs[1]
        return (a, b), lambda t: a * t ** b
    except:
        return None


# Метрики качества
def calculate_metrics(x, y, f):
    n = len(x)
    y_pred = [f(xi) for xi in x]
    residuals = [(y[i] - y_pred[i]) for i in range(n)]
    S = sum(resid ** 2 for resid in residuals)
    sigma = (S / n) ** 0.5
    y_mean = mean(y)
    St = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / St if St != 0 else 0
    r = correlation_coefficient(y, y_pred)
    return sigma, R2, S, r


# GUI приложение
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Аппроксимация функций")
        self.x, self.y = [], []

        self.frame = ttk.Frame(root, padding=10)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)

        self.text = tk.Text(root, width=70, height=40, font=("Consolas", 10))
        self.text.pack(side=tk.RIGHT, padx=10, pady=10)

        # Кнопки
        ttk.Button(self.frame, text="Загрузить из файла", command=self.load_file).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame, text="Ручной ввод", command=self.manual_input).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame, text="Аппроксимировать", command=self.approximate).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame, text="Сохранить результат", command=self.save_result).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame, text="Очистить консоль", command=self.clear_console).pack(fill=tk.X, pady=5)

        # График
        self.figure = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(pady=10)

    def clear_console(self):
        self.text.delete(1.0, tk.END)  # Очищаем всё содержимое текстового поля

    def load_file(self):
        filename = filedialog.askopenfilename(title="Выберите файл", filetypes=[("Text files", "*.txt")])
        if not filename:
            return
        with open(filename, "r", encoding='utf-8') as file:
            lines = file.readlines()
            self.x, self.y = [], []
            self.text.insert(tk.END, "Загруженные точки:\n")  # Заголовок для отображения точек
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x_val = parse_number(parts[0])
                    y_val = parse_number(parts[1])
                    if x_val is not None and y_val is not None:
                        self.x.append(x_val)
                        self.y.append(y_val)
                        # Добавляем точку в консоль (в текстовое поле)
                        self.text.insert(tk.END, f"({x_val}, {y_val})\n")
                    else:
                        messagebox.showwarning("Предупреждение", f"Не удалось распознать числа в строке: {line}")
            self.text.insert(tk.END, f"Загружено {len(self.x)} точек\n")

    def manual_input(self):
        input_window = tk.Toplevel(self.root)
        input_window.title("Ручной ввод точек")
        tk.Label(input_window, text="Введите точки через пробел: (x1 y1 x2 y2 ...)").pack()
        entry = tk.Entry(input_window, width=50)
        entry.pack()

        def save():
            try:
                values_str = entry.get().split()
                values = []
                for val in values_str:
                    num = parse_number(val)
                    if num is None:
                        raise ValueError(f"Неверный формат числа: {val}")
                    values.append(num)

                if len(values) % 2 != 0:
                    raise ValueError("Количество чисел должно быть четным (x1 y1 x2 y2 ...)")

                self.x = values[::2]
                self.y = values[1::2]
                self.text.insert(tk.END, f"Введено {len(self.x)} точек\n")
                input_window.destroy()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        ttk.Button(input_window, text="Сохранить", command=save).pack(pady=5)

    def approximate(self):
        if not self.x or not self.y:
            messagebox.showerror("Ошибка", "Нет данных!")
            return
        self.text.delete(1.0, tk.END)  # Очищаем текстовое поле перед выводом результатов
        self.ax.clear()
        self.ax.scatter(self.x, self.y, color="black", label="Данные")

        methods = [
            ("Линейная", linear_approximation, "f(x) = a + b * xi"),
            ("Полиноминальная 2-й степени", quadratic_approximation, "f(x) = a + b * xi + c * xi²"),
            ("Полиноминальная 3-й степени", cubic_approximation, "f(x) = a + b * xi + c * xi² + d * xi³"),
            ("Экспоненциальная", exponential_approximation, "f(x) = a * exp(b * xi)"),
            ("Логарифмическая", logarithmic_approximation, "f(x) = a + b * log(xi)"),
            ("Степенная", power_approximation, "f(x) = a * xi^b"),
        ]
        colors = ["red", "green", "blue", "orange", "purple", "cyan"]
        t = [min(self.x) + i * (max(self.x) - min(self.x)) / 500 for i in range(500)]

        best_R2 = -float("inf")
        best_method = ""

        self.result_text = ""

        for (name, method, formula), color in zip(methods, colors):
            result = method(self.x, self.y)
            if result is None:
                continue
            coeffs, f = result
            sigma, R2, S, r = calculate_metrics(self.x, self.y, f)
            y_fit = [f(i) for i in t]
            self.ax.plot(t, y_fit, label=name, color=color)

            self.result_text += f"{name} функция:\n"
            self.result_text += f"*  Функция: {formula}\n"
            self.result_text += f"*  Коэффициенты: {[round(c, 4) for c in coeffs]}\n"
            self.result_text += f"*  Среднеквадратичное отклонение: σ = {sigma:.5f}\n"
            self.result_text += f"*  Коэффициент детерминации: R^2 = {R2:.5f}\n"
            self.result_text += f"*  Мера отклонения: S = {S:.5f}\n"
            self.result_text += f"*  Коэффициент корреляции Пирсона: r = {r:.16f}\n"
            self.result_text += "-" * 30 + "\n\n"

            if R2 > best_R2:
                best_R2 = R2
                best_method = name

        self.result_text += f"Лучшая функция приближения: {best_method}\n"
        self.text.insert(tk.END, self.result_text)

        self.ax.legend()
        self.ax.grid()
        self.canvas.draw()

    def save_result(self):
        if not hasattr(self, "result_text") or not self.result_text:
            messagebox.showinfo("Информация", "Нет результатов для сохранения.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.result_text)
            messagebox.showinfo("Успех", f"Результат сохранён в {filename}")


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()