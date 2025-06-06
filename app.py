from flask import Flask, render_template, request, jsonify,  send_from_directory
import numpy as np

app = Flask(__name__)

def matrix_to_str(M):
    return '\n'.join(['\t'.join(f"{v:.4g}" for v in row) for row in M])


# ----------Методы решения----------

# ---------- Метод Гаусса с сохранением шагов ----------
def solve_gauss(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    full_matrix = np.hstack((A, b.reshape(-1, 1)))
    n = len(full_matrix)
    steps.append("Исходная расширенная матрица [A|b]:\n" + matrix_to_str(full_matrix))

    for i in range(n):
        steps.append(f"\nЭтап {i + 1}: Исключение x{i + 1}")
        max_row = i
        for j in range(i, n):
            if abs(full_matrix[j][i]) > abs(full_matrix[max_row][i]):
                max_row = j
        steps.append(f"Выбран ведущий элемент: {full_matrix[max_row][i]:.2f} (строка {max_row + 1})")
        if max_row != i:
            full_matrix[[i, max_row]] = full_matrix[[max_row, i]]
            steps.append(f"Перестановка строк {i + 1} и {max_row + 1}")
            steps.append("Матрица после перестановки:\n" + matrix_to_str(full_matrix))

        leading_element = full_matrix[i][i]
        if abs(leading_element) < 1e-10:
            steps.append("Нулевой ведущий элемент — система несовместна или имеет бесконечно много решений.")
            return None, steps

        for j in range(i + 1, n):
            factor = full_matrix[j][i] / leading_element
            full_matrix[j] -= factor * full_matrix[i]
            steps.append(f"Строка {j + 1} = Строка {j + 1} - ({factor:.2f}) × Строка {i + 1}")
            steps.append("Результат:\n" + matrix_to_str(full_matrix))

    x = np.zeros(n)
    steps.append("\nОбратный ход:")
    for i in range(n - 1, -1, -1):
        known_sum = sum(full_matrix[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (full_matrix[i][-1] - known_sum) / full_matrix[i][i]
        steps.append(f"x{i + 1} = ({full_matrix[i][-1]:.2f} - {known_sum:.2f}) / {full_matrix[i][i]:.2f} = {x[i]:.2f}")

    return x.tolist(), steps



def solve_gauss_jordan(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    full_matrix = np.hstack((A, b.reshape(-1, 1)))
    n = len(full_matrix)
    steps.append("Исходная расширенная матрица [A|b]:\n" + matrix_to_str(full_matrix))

    for i in range(n):
        steps.append(f"\nЭтап {i + 1}: Обнуление столбца x{i + 1}")
        max_row = i
        for j in range(i, n):
            if abs(full_matrix[j][i]) > abs(full_matrix[max_row][i]):
                max_row = j
        steps.append(f"Выбран ведущий элемент: {full_matrix[max_row][i]:.2f} (строка {max_row + 1})")
        if max_row != i:
            full_matrix[[i, max_row]] = full_matrix[[max_row, i]]
            steps.append(f"Перестановка строк {i + 1} и {max_row + 1}")
            steps.append("Матрица после перестановки:\n" + matrix_to_str(full_matrix))

        leading_element = full_matrix[i][i]
        if abs(leading_element) < 1e-10:
            steps.append("Нулевой ведущий элемент — система несовместна или имеет бесконечно много решений.")
            return None, steps

        full_matrix[i] = full_matrix[i] / leading_element
        steps.append(f"Нормализация строки {i + 1} (делим на {leading_element:.2f})")
        steps.append("Результат:\n" + matrix_to_str(full_matrix))

        for j in range(n):
            if j != i:
                factor = full_matrix[j][i]
                full_matrix[j] -= factor * full_matrix[i]
                steps.append(f"Строка {j + 1} = Строка {j + 1} - ({factor:.2f}) × Строка {i + 1}")
                steps.append("Результат:\n" + matrix_to_str(full_matrix))

    x = full_matrix[:, -1]
    steps.append("\nИтог: получена единичная матрица, решения в последнем столбце.")
    for i in range(n):
        steps.append(f"x{i + 1} = {x[i]:.2f}")

    return x.tolist(), steps


# def get_det(m, steps, level=0):
#     indent = "  " * level
#     if len(m) == 1:
#         steps.append(f"{indent}Определитель 1x1: {m[0, 0]:.2f}")
#         return m[0, 0]
#     elif len(m) == 2:
#         det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
#         steps.append(
#             f"{indent}Определитель 2x2: ({m[0, 0]:.2f}×{m[1, 1]:.2f}) - ({m[0, 1]:.2f}×{m[1, 0]:.2f}) = {det:.2f}")
#         return det
#     else:
#         D = 0
#         steps.append(f"{indent}Разложение по 1-й строке матрицы {m.shape}:")
#         for i in range(m.shape[1]):
#             steps.append(f"{indent}≡ Элемент A[0,{i}] = {m[0, i]:.2f}")
#             minor = np.delete(np.delete(m, 0, axis=0), i, axis=1)
#             steps.append(f"{indent}≡ Минор для элемента {m[0, i]:.2f}:")
#             steps.append(f"{indent}" + matrix_to_str(minor))
#             minor_det = get_det(minor, steps, level + 1)
#             sign = (-1) ** i
#             term = sign * m[0, i] * minor_det
#             steps.append(f"{indent}≡ (-1)^{i} × {m[0, i]:.2f} × {minor_det:.2f} = {term:.2f}")
#             D += term
#             steps.append(f"{indent}≡ Промежуточный определитель: {D:.2f}")
#         return D


def solve_cramer(A, b):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    steps.append("Исходная матрица коэффициентов A:")
    steps.append(matrix_to_str(A))
    steps.append("Вектор свободных членов b:")
    steps.append('\t'.join(f"{num:.4g}" for num in b))

    det_A = np.linalg.det(A)
    steps.append(f"\nВычисляем определитель det(A): {det_A:.6g}")

    if abs(det_A) < 1e-12:
        steps.append("Определитель близок к нулю — система либо несовместна, либо имеет бесконечно много решений.")
        return None, steps

    x = []
    for i in range(n):
        steps.append(f"\nЗамена {i + 1}-го столбца матрицы A на вектор b для вычисления det(A{i + 1}):")
        A_copy = A.copy()
        A_copy[:, i] = b
        steps.append(matrix_to_str(A_copy))

        det_A_copy = np.linalg.det(A_copy)
        steps.append(f"Вычисляем определитель det(A{i + 1}): {det_A_copy:.6g}")

        xi = det_A_copy / det_A
        steps.append(f"Вычисляем x{i + 1} = det(A{i + 1}) / det(A) = {det_A_copy:.6g} / {det_A:.6g} = {xi:.6g}")
        x.append(xi)

    steps.append("\nИтог: решение системы (вектор x):")
    steps.append('\t'.join(f"x{i+1} = {val:.6g}" for i, val in enumerate(x)))

    return x, steps



def solve_iteration(A, b, eps=1e-8, max_iter=1000, iterations_count=None):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    # Проверка диагонального преобладания
    diag_dom = all(abs(A[i, i]) > sum(abs(A[i, j]) for j in range(n) if j != i) for i in range(n))
    steps.append("Проверка на диагональное преобладание:")
    steps.append("Да" if diag_dom else "Нет")

    # Построение α и β
    alpha = -A / A.diagonal().reshape(-1, 1)
    np.fill_diagonal(alpha, 0)
    beta = b / A.diagonal()

    # Проверка нормы матрицы α (по строкам)
    alpha_norm = np.max(np.sum(np.abs(alpha), axis=1))
    steps.append(f"Норма матрицы α (макс. сумма по строкам): {alpha_norm:.6f}")
    if not diag_dom and alpha_norm >= 1:
        steps.append("Система не удовлетворяет условиям сходимости метода итераций (нет диагонального преобладания и ||α|| ≥ 1).")
        return None, steps

    steps.append("Матрица A:")
    steps.append(matrix_to_str(A))
    steps.append("Вектор b:")
    steps.append('\t'.join(f"{val:.6g}" for val in b))
    steps.append("\nМатрица α (α = -A / diag(A)):")
    steps.append(matrix_to_str(alpha))
    steps.append("Вектор β (β = b / diag(A)):")
    steps.append('\t'.join(f"{val:.6g}" for val in beta))

    x_prev = beta.copy()
    steps.append("\nНачальное приближение (x₀ = β):")
    steps.append('\t'.join(f"{val:.6g}" for val in x_prev))

    results_list = [x_prev.copy()]
    iterations = 0

    target_iter = iterations_count or max_iter
    for iteration in range(1, target_iter + 1):
        x = beta + np.dot(alpha, x_prev)
        delta = np.max(np.abs(x - x_prev))

        steps.append(f"\nИтерация {iteration}:")
        # steps.append("x⁽ⁿ⁺¹⁾ = β + α * x⁽ⁿ⁾")
        steps.append("Результат:")
        steps.append('\t'.join(f"{val:.8f}" for val in x))
        steps.append(f"Максимальное изменение: {delta:.2e}")

        results_list.append(x.copy())
        iterations += 1

        if iterations_count is None and delta < eps:
            steps.append(f"\nДостигнута требуемая точность: Δ < ε ({delta:.2e} < {eps})")
            break
        x_prev = x
    else:
        steps.append(f"\nПревышено максимальное число итераций ({iterations_count})")

    steps.append(f"\nЧисло итераций: {iterations}")
    return x.tolist(), steps



def solve_seidel(A, b, eps=1e-8, max_iter=1000, iterations_count=None):
    steps = []
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    # Проверка диагонального преобладания
    diag_dom = all(abs(A[i, i]) > sum(abs(A[i, j]) for j in range(n) if j != i) for i in range(n))
    steps.append("Проверка на диагональное преобладание:")
    steps.append("Да" if diag_dom else "Нет")

    C = A.T @ A
    d = A.T @ b

    steps.append("Матрица C = AᵀA:")
    steps.append(matrix_to_str(C))
    steps.append("Вектор d = Aᵀb:")
    steps.append('\t'.join(f"{val:.6g}" for val in d))

    alpha = -C / C.diagonal().reshape(-1, 1)
    np.fill_diagonal(alpha, 0)
    beta = d / C.diagonal()

    # Норма α для Сейделя
    alpha_norm = np.max(np.sum(np.abs(alpha), axis=1))
    steps.append(f"Норма матрицы α (по строкам): {alpha_norm:.6f}")
    if not diag_dom and alpha_norm >= 1:
        steps.append("Система не удовлетворяет условиям сходимости метода Зейделя (нет диагонального преобладания и ||α|| ≥ 1).")
        return None, steps

    steps.append("\nМатрица α (α = -C / diag(C), без диагонали):")
    steps.append(matrix_to_str(alpha))
    steps.append("Вектор β (β = d / diag(C)):")
    steps.append('\t'.join(f"{val:.6g}" for val in beta))

    L = np.tril(alpha)
    U = np.triu(alpha)
    E = np.eye(n)

    x_prev = beta.copy()
    steps.append("\nНачальное приближение (x₀ = β):")
    steps.append('\t'.join(f"{val:.6g}" for val in x_prev))

    iterations = 0
    results_list = [x_prev.copy()]

    target_iter = iterations_count or max_iter
    for iteration in range(1, target_iter + 1):
        inv_matrix = np.linalg.inv(E - L)
        x = inv_matrix @ (U @ x_prev + beta)

        delta = np.max(np.abs(x - x_prev))
        iterations += 1
        results_list.append(x.copy())

        steps.append(f"\nИтерация {iteration}:")
        # steps.append(f"x⁽ⁿ⁺¹⁾ = (E - L)⁻¹ (U x⁽ⁿ⁾ + β)")
        steps.append("Результат:")
        steps.append('\t'.join(f"{val:.8f}" for val in x))
        steps.append(f"Максимальное изменение: {delta:.2e}")

        if iterations_count is None and delta < eps:
            steps.append(f"\nДостигнута требуемая точность: Δ < ε ({delta:.2e} < {eps})")
            break
        x_prev = x
    else:
        steps.append(f"\nПревышено максимальное число итераций ({iterations_count})")

    steps.append(f"\nЧисло итераций: {iterations}")
    steps.append("Решение:")
    for i, val in enumerate(x):
        steps.append(f"x{i + 1} = {val:.8f}")

    return x.tolist(), steps



# ----------Flask-маршруты----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory('img', filename)

@app.route('/solve', methods=['POST'])
def solve():
    try:
        A, b = [], []
        method = request.form.get('method', 'gauss')  # по умолчанию метод Гаусса
        size = int(request.form.get('size', 3))

        iter_count_str = request.form.get('iterations')
        iterations_count = int(iter_count_str) if iter_count_str and iter_count_str.isdigit() else None

        for i in range(size):
            row = []
            for j in range(size):
                row.append(float(request.form[f'a{i}{j}']))
            A.append(row)
            b.append(float(request.form[f'b{i}']))

        # Вызов нужного метода
        if method == 'gauss':
            x, steps = solve_gauss(A, b)
        elif method == 'gauss_jordan':
            x, steps = solve_gauss_jordan(A, b)
        elif method == 'cramer':
            x, steps = solve_cramer(A, b)
        elif method == 'iteration':
            x, steps = solve_iteration(A, b, iterations_count=iterations_count)
        elif method == 'seidel':
            x, steps = solve_seidel(A, b, iterations_count=iterations_count)
        else:
            return jsonify({'x': None, 'steps': [f"Метод '{method}' ещё не реализован."]})

        return jsonify({'x': x, 'steps': steps})

    except Exception as e:
        return jsonify({'x': None, 'steps': [f"Ошибка: {str(e)}"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=55555, debug=True)