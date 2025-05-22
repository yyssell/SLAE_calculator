from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# ---------- Методы решения ----------

# ---------- Метод Гаусса с сохранением шагов ----------
def solve_gauss(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    augmented = np.hstack((A, b.reshape(-1, 1)))
    n = len(augmented)
    steps.append("1. Исходная расширенная матрица [[A|b]]:\n" + matrix_to_str(augmented))
    # Прямой ход - последовательное исключение переменных
    for i in range(n):
        steps.append(f"\n--- Этап {i + 1}: Исключение x_{i + 1} ---")
        max_row = i
        for j in range(i, n):
            if abs(augmented[j][i]) > abs(augmented[max_row][i]):
                max_row = j
        steps.append(f"2. Ведущий элемент: {augmented[max_row][i]:.2f} (строка {max_row + 1})")
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            steps.append(f"3. Переставлены строки {i + 1} ↔ {max_row + 1} для численной устойчивости")
            steps.append("   Текущая матрица:\n" + matrix_to_str(augmented))
        pivot = augmented[i][i]
        if abs(pivot) < 1e-10:
            steps.append("4. Система несовместна/неопределена (нулевой ведущий элемент)")
            return None, steps
        steps.append(f"5. Исключаем x_{i + 1} из уравнений с {i + 2} по {n}:")
        for j in range(i + 1, n):
            factor = augmented[j][i] / pivot
            augmented[j] -= factor * augmented[i]
            steps.append(f"   • Строка {j + 1} = Строка {j + 1} - ({factor:.2f}) × Строка {i + 1}")
            steps.append(f"     Результат: {augmented[j]}")

    # Обратный ход - нахождение значений переменных [[4]]
    x = np.zeros(n)
    steps.append("\n--- Обратная подстановка ---")
    for i in range(n - 1, -1, -1):
        # Сумма известных членов: Σ(a_ij * x_j) для j > i [[5]]
        known_sum = sum(augmented[i][k] * x[k] for k in range(i + 1, n))
        equation = f"{augmented[i][-1]:.2f} - {known_sum:.2f}"
        x[i] = (augmented[i][-1] - known_sum) / augmented[i][i]
        steps.append(f"   • x_{i + 1} = ({equation}) / {augmented[i][i]:.2f} = {x[i]:.2f}")

    return x, steps


def matrix_to_str(matrix):
    return '\n'.join([' '.join([f"{num:8.2f}" for num in row]) for row in matrix])

def solve_gauss_jordan(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    augmented = np.hstack((A, b.reshape(-1, 1)))
    n = len(augmented)

    steps.append("1. Исходная расширенная матрица [[A|b]]:\n" + matrix_to_str(augmented))

    # Прямой ход с обнулением как вниз, так и вверх от ведущего элемента
    for i in range(n):
        steps.append(f"\n--- Этап {i + 1}: Работа с ведущим элементом в строке {i + 1} ---")

        # 1) Выбор ведущего элемента
        max_row = i
        for j in range(i, n):
            if abs(augmented[j][i]) > abs(augmented[max_row][i]):
                max_row = j
        pivot = augmented[max_row][i]
        steps.append(f"  Ведущий элемент: {pivot:.2f} (строка {max_row + 1})")
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            steps.append(f"  Переставлены строки {i + 1} ↔ {max_row + 1}")
            steps.append("  Текущая матрица:\n" + matrix_to_str(augmented))
        pivot = augmented[i][i]
        if abs(pivot) < 1e-12:
            steps.append("  Нулевой ведущий элемент — система несовместна или вырождена")
            return None, steps
        augmented[i] = augmented[i] / pivot
        steps.append(f"  Нормализовали строку {i + 1}, разделив на {pivot:.2f}")
        steps.append("  Результат:\n" + matrix_to_str(augmented))

        # 4) Обнуление всех остальных элементов в столбце i
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                augmented[j] -= factor * augmented[i]
                steps.append(f"  Вычли из строки {j + 1} строку {i + 1} × {factor:.2f}")
                steps.append("  Текущая матрица:\n" + matrix_to_str(augmented))

    # После всех шагов в расширенной матрице останется [I | x]
    x = augmented[:, -1]
    steps.append("\n--- Готово: получена приведённая к единичной форме матрица, решения в последнем столбце ---")
    steps.append("Решение: x = [" + ", ".join(f"{xi:.2f}" for xi in x) + "]")

    return x, steps


def get_det(m, steps, level=0):
    indent = "  " * level
    if len(m) == 1:
        steps.append(f"{indent}Определитель 1x1: {m[0, 0]:.2f}")
        return m[0, 0]
    elif len(m) == 2:
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        steps.append(
            f"{indent}Определитель 2x2: ({m[0, 0]:.2f}×{m[1, 1]:.2f}) - ({m[0, 1]:.2f}×{m[1, 0]:.2f}) = {det:.2f}")
        return det
    else:
        D = 0
        steps.append(f"{indent}Разложение по 1-й строке матрицы {m.shape}:")
        for i in range(m.shape[1]):
            steps.append(f"{indent}≡ Элемент A[0,{i}] = {m[0, i]:.2f}")
            minor = np.delete(np.delete(m, 0, axis=0), i, axis=1)
            steps.append(f"{indent}≡ Минор для элемента {m[0, i]:.2f}:")
            steps.append(f"{indent}" + matrix_to_str(minor))
            minor_det = get_det(minor, steps, level + 1)
            sign = (-1) ** i
            term = sign * m[0, i] * minor_det
            steps.append(f"{indent}≡ (-1)^{i} × {m[0, i]:.2f} × {minor_det:.2f} = {term:.2f}")
            D += term
            steps.append(f"{indent}≡ Промежуточный определитель: {D:.2f}")
        return D


def solve_cramer(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    steps.append("1. Формируем матрицу системы и вектор свободных членов:")
    steps.append("Матрица A:\n" + matrix_to_str(A))
    steps.append("Вектор b:\n" + "\n".join(f"{num:8.2f}" for num in b))

    steps.append("\n2. Вычисляем главный определитель D:")
    D = get_det(A, steps)
    steps.append(f"Главный определитель D = {D:.2f}")

    if abs(D) < 1e-10:
        steps.append("Система несовместна или имеет бесконечно много решений (D = 0)")
        return None, steps

    x = []
    for i in range(n):
        steps.append(f"\n--- Вычисление x_{i + 1} ---")
        steps.append(f"3.1 Заменяем {i + 1}-й столбец на вектор b:")
        A_i = A.copy()
        A_i[:, i] = b
        steps.append("Матрица A_" + str(i + 1) + ":\n" + matrix_to_str(A_i))
        steps.append(f"3.2 Вычисляем определитель D_{i + 1}:")
        D_i = get_det(A_i, steps)
        steps.append(f"D_{i + 1} = {D_i:.2f}")
        x_i = D_i / D
        x.append(x_i)
        steps.append(f"3.3 x_{i + 1} = {D_i:.2f} / {D:.2f} = {x_i:.2f}")

    steps.append("\n4. Итоговое решение:")
    return x, steps


def solve_iteration(A, b, eps=1e-6, max_iter=1000):
    steps = []
    n = len(A)
    alpha = -A / A.diagonal().reshape(n, 1)
    np.fill_diagonal(alpha, 0)
    beta = b / A.diagonal()
    steps.append("1. Формируем матрицу α и вектор β:")
    steps.append(f"α:\n{matrix_to_str(alpha)}")
    steps.append(f"β:\n{np.array_str(beta, precision=4)}")
    x_prev = np.zeros_like(beta)  # Начальное приближение
    results = [x_prev.copy()]
    steps.append(f"\n2. Начальное приближение x₀: {x_prev}")
    for i in range(max_iter):
        x = beta + alpha @ x_prev
        results.append(x.copy())
        steps.append(f"\n--- Итерация {i + 1} ---")
        steps.append(f"3. Вычисляем x_{i + 1} = β + α × x_{i}:")
        steps.append(f"   x_{i + 1} = {beta} + {matrix_to_str(alpha)} × {x_prev}")
        steps.append(f"          = {x}")
        relative_error = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        steps.append(f"4. Относительная погрешность: {relative_error:.2e}")
        if relative_error < eps:
            steps.append(f"5. Погрешность меньше {eps} → сходимость достигнута")
            return x, i + 1, steps
        x_prev = x
    steps.append(f"5. Достигнуто максимальное количество итераций ({max_iter})")
    return x, max_iter, steps


def solve_seidel(A, b, eps=1e-6, max_iter=1000):
    steps = []
    n = len(A)
    # Формируем матрицу системы C = A^T*A и вектор d = A^T*b
    C = A.T @ A
    d = A.T @ b
    steps.append(f"1. Формируем нормальные уравнения: C = A^T*A, d = A^T*b [[4]]")
    steps.append(f"   C:\n{matrix_to_str(C)}")
    steps.append(f"   d:\n{np.array_str(d, precision=4)}")
    # Формируем матрицу alpha и вектор beta
    alpha = -C / C.diagonal().reshape(n, 1)
    np.fill_diagonal(alpha, 0)
    beta = d / C.diagonal()
    steps.append(f"\n2. Формируем α = -C/D_ii и β = d/D_ii [[7]]")
    steps.append(f"   α:\n{matrix_to_str(alpha)}")
    steps.append(f"   β:\n{np.array_str(beta, precision=4)}")
    # Разделяем на нижнюю и верхнюю треугольные матрицы
    L = np.tril(alpha)
    U = np.triu(alpha)
    steps.append(f"\n3. Разделяем α на L (нижняя) и U (верхняя) [[2]]")
    steps.append(f"   L:\n{matrix_to_str(L)}")
    steps.append(f"   U:\n{matrix_to_str(U)}")
    # Формируем матрицу перехода и вектор смещения
    inv_E_L = np.linalg.inv(np.eye(n) - L)
    P = inv_E_L @ U
    Q = inv_E_L @ beta
    steps.append(f"\n4. Вычисляем P = (E-L)^-1*U и Q = (E-L)^-1*β [[8]]")
    x_prev = np.zeros_like(beta)  # Начальное приближение
    steps.append(f"\n5. Начальное приближение x₀: {x_prev}")
    for i in range(max_iter):
        x = P @ x_prev + Q
        steps.append(f"\n--- Итерация {i + 1} ---")
        steps.append(f"6. Вычисляем x_{i + 1} = P × x_{i} + Q:")
        steps.append(f"   x_{i + 1} = {matrix_to_str(P)} × {x_prev} + {Q}")
        steps.append(f"            = {x}")
        # Проверка сходимости
        relative_error = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        steps.append(f"7. Относительная погрешность: {relative_error:.2e} [[4]]")
        if relative_error < eps:
            steps.append(f"8. Погрешность меньше {eps} → сходимость достигнута")
            return x, i + 1, steps
        x_prev = x
    steps.append(f"8. Достигнуто максимальное количество итераций ({max_iter}) [[6]]")
    return x, max_iter, steps


# ---------- Flask-маршруты ----------

@app.route('/', methods=['GET', 'POST'])
def index():
    steps, x, method = [], None, None
    if request.method == 'POST':
        try:
            # Парсинг матрицы
            matrix = []
            for line in request.form['matrix'].split('\n'):
                row = [float(num) for num in line.split()]
                matrix.append(row)
            # Парсинг вектора
            vector = [float(num) for num in request.form['vector'].split()]
            method = request.form['method']
            # Выбор метода решения
            solver_map = {
                'gauss': solve_gauss,
                'gauss_jordan': solve_gauss_jordan,
                'cramer': solve_cramer,
                'iteration': solve_iteration,
                'seidel': solve_seidel
            }
            if method in ['iteration', 'seidel']:
                x, _, steps = solver_map[method](matrix, vector)
            else:
                x, steps = solver_map[method](matrix, vector)
        except Exception as e:
            steps = [f"Ошибка: {str(e)}"]
    return render_template('index.html', steps=steps, x=x, method=method)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=44444, debug=True)