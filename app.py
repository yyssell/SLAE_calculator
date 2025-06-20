from flask import Flask, render_template, request, jsonify,  send_from_directory, make_response
import numpy as np
import csv
from io import StringIO

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


def solve_iteration(A, b, eps=None, max_iter=1000):
    steps = []

    # Преобразуем входные данные в массивы NumPy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    rows, cols = A.shape

    # Формируем расширенную матрицу (a | b)
    matrix = np.hstack((A, b.reshape(-1, 1)))

    # Преобразуем матрицу
    for i in range(rows):
        diag = A[i, i]
        steps.append(f"\nПреобразование строки {i + 1}:")
        steps.append(f"Делим элементы строки на диагональный элемент a[{i + 1},{i + 1}] = {diag:.2f}")
        for j in range(cols):
            if j != i:
                matrix[i, j] = -A[i, j] / diag
                steps.append(f"  a[{i + 1},{j + 1}] = -{A[i, j]:.2f} / {diag:.2f} = {matrix[i, j]:.2f}")
        matrix[i, cols] = b[i] / diag
        matrix[i, i] = 0
        steps.append(f"  b[{i + 1}] = {b[i]:.2f} / {diag:.2f} = {matrix[i, cols]:.2f}")

    sum_of_squares = np.sqrt(np.sum(matrix[:, :cols] ** 2))
    if sum_of_squares > 1:
        steps.append("Условие сходимости не выполняется для метода простых итераций")
        return None, steps
    else:
        if eps is None:
            eps = 1e-6

        current = matrix[:, cols].copy()
        distance = eps + 1
        iter_count = 0
        result = np.zeros(rows)

        steps.append("\nНачальное приближение:")
        steps.append(str(current))

        while distance > eps and (max_iter is None or iter_count < max_iter):
            steps.append(f"\nИтерация {iter_count + 1}:")
            for i in range(rows):
                steps.append(f"Обновление x[{i + 1}]:")
                steps.append(f"  x[{i + 1}] = b[{i + 1}] + (a[{i + 1},j] * x[j])")
                # В методах solve_iteration и solve_seidel:
                steps.append(
                    f"  x[{i + 1}] = {matrix[i, cols]:.2f} + ("
                    f"{[round(float(x), 2) for x in matrix[i, :cols]]} * "
                    f"{[round(float(x), 2) for x in current]})"
                )
                result[i] = matrix[i, cols] + np.dot(matrix[i, :cols], current)
                steps.append(f"  x[{i + 1}] = {result[i]:.2f}")
            distance = np.linalg.norm(result - current)
            steps.append(f"Расстояние между текущим и предыдущим приближением: {distance:.10f}")
            current = result.copy()
            iter_count += 1

        steps.append(f"\nРешение методом простых итераций после {iter_count} итераций:")
        steps.append(str(result))


    return result.tolist(), steps


def solve_seidel(A, b, eps=None, max_iter=1000):
    steps = []
    # Преобразуем входные данные в массивы NumPy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    rows, cols = A.shape
    # Формируем расширенную матрицу (a | b)
    matrix = np.hstack((A, b.reshape(-1, 1)))

    sum_of_squares = 0
    current = b / np.diag(A)  # начальное приближение (b[i]/a[i,i])

    # Преобразуем матрицу в итерационную форму
    for i in range(rows):
        diag = A[i, i]
        steps.append(f"Преобразование строки {i + 1}:")
        steps.append(f"Делим элементы строки на диагональный элемент a[{i + 1},{i + 1}] = {diag:.2f}")
        for j in range(cols):
            if j != i:
                matrix[i, j] = -A[i, j] / diag
                sum_of_squares += matrix[i, j] ** 2
                steps.append(f"a[{i + 1},{j + 1}] = -{A[i, j]:.2f} / {diag:.2f} = {matrix[i, j]:.2f}")
        matrix[i, cols] = b[i] / diag
        matrix[i, i] = 0
        steps.append(f"b[{i + 1}] = {b[i]:.2f} / {diag:.2f} = {matrix[i, cols]:.2f}")
    if np.sqrt(sum_of_squares) > 1:
        steps.append("Условие сходимости не выполняется для метода Зейделя")
        return None, steps
    if np.sqrt(sum_of_squares) < 1:
        if eps is None:
            eps = 1e-6

        distance = 2 * eps
        result = current.copy()
        iter_count = 0
        steps.append(f"Начальное приближение: {str(current)}")
        # steps.append()

        while distance > eps and (max_iter is None or iter_count < max_iter):
            distance = 0
            steps.append(f"Итерация {iter_count + 1}:")

            for i in range(rows):
                old_value = current[i]

                result[i] = matrix[i, cols] + np.dot(matrix[i, :cols], current)
                steps.append(f"Обновление x[{i + 1}]:" + f"\nx[{i + 1}] = b[{i + 1}] + (a[{i + 1},j] * x[j])" + f"\nx[{i + 1}] = {matrix[i, cols]:.2f} + ("
                    f"{[round(float(x), 2) for x in matrix[i, :cols]]} * "
                    f"{[round(float(x), 2) for x in current]})" + f"\nx[{i + 1}] = {result[i]:.2f}")

                distance += (result[i] - old_value) ** 2
                current[i] = result[i]  # Обновляем текущее значение сразу (особенность метода Зейделя)
            distance = np.sqrt(distance)
            steps.append(f"Расстояние между текущим и предыдущим приближением: {distance:.10f}")
            iter_count += 1

        steps.append(f"Достигнута требуемоя точность\nРешение методом Зейделя после {iter_count} итераций:")
        steps.append(str(result))


    return result.tolist(), steps


# ----------Flask-маршруты----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory('img', filename)


@app.route('/generate_txt', methods=['POST'])
def generate_txt():
    try:
        data = request.json
        content = "Результат решения СЛАУ:\n"
        content += f"x = [{', '.join(f'{val:.4f}' for val in data['x'])}]\n\n"
        content += "Шаги решения:\n"
        content += "\n".join(data['steps'])
        response = make_response(content)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename=result.txt'
        return response
    except Exception as e:
        return f"Ошибка: {str(e)}", 500


@app.route('/export_csv', methods=['POST'])
def export_csv():
    data = request.json
    A = data.get('A', [])
    b = data.get('b', [])
    x = data.get('x', [])
    steps = data.get('steps', [])
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    if A:
        headers = [f'A{i + 1}' for i in range(len(A[0]))] + ['b']
        csv_writer.writerow(headers)
        for row in A:
            csv_row = list(row) + [b[A.index(row)]]
            csv_writer.writerow(csv_row)
        csv_writer.writerow([])
    csv_writer.writerow(['Шаги решения:'])
    for step in steps:
        cleaned_step = step.replace('\t', '    ').replace('"', '""')
        csv_writer.writerow([cleaned_step])
    csv_writer.writerow(['Решение:'])
    csv_writer.writerow([f'x = {", ".join(map(str, x))}'])
    csv_content = csv_buffer.getvalue()
    response = make_response(f'\ufeff{csv_content}')  # Добавляем BOM
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = 'attachment; filename=solution.csv'
    return response

@app.route('/solve', methods=['POST'])
def solve():
    try:
        A, b = [], []
        method = request.form.get('method', 'gauss')  # по умолчанию метод Гаусса
        size = int(request.form.get('size', 3))
        precision_count_str = request.form.get('precision')
        precision = float(precision_count_str) if precision_count_str else None
        for i in range(size):
            row = []
            for j in range(size):
                row.append(float(request.form[f'a{i}{j}']))
            A.append(row)
            b.append(float(request.form[f'b{i}']))
        # Вызов нужного метода
        if method == 'gauss':
            x, steps = solve_gauss(A, b)
            solution_data = {
                'A': A,
                'b': b,
                'x': x,
                'steps': steps
            }
        elif method == 'gauss_jordan':
            x, steps = solve_gauss_jordan(A, b)
            solution_data = {
                'A': A,
                'b': b,
                'x': x,
                'steps': steps
            }
        elif method == 'cramer':
            x, steps = solve_cramer(A, b)
            solution_data = {
                'A': A,
                'b': b,
                'x': x,
                'steps': steps
            }
        elif method == 'iteration':
            x, steps = solve_iteration(A, b, eps=precision)
            solution_data = {
                'A': A,
                'b': b,
                'x': x,
                'steps': steps
            }
        elif method == 'seidel':
            x, steps = solve_seidel(A, b, eps=precision)
            solution_data = {
                'A': A,
                'b': b,
                'x': x,
                'steps': steps
            }
        else:
            return jsonify({'x': None, 'steps': [f"Метод '{method}' ещё не реализован."]})

        return jsonify(solution_data)

    except Exception as e:
        return jsonify(solution_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=55555, debug=True)