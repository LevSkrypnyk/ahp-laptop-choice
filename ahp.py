"""
ahp.py - модуль для розрахунків методом аналізу ієрархій (МАІ, метод Сааті).

Тут зібрані всі "математичні" функції: побудова матриці парних порівнянь,
пошук вектора пріоритетів (власний вектор), перевірка узгодженості (CI, CR)
і синтез глобальних пріоритетів. Модуль не прив'язаний до ноутбуків,
тому його можна використати для будь-якої іншої задачі вибору.
"""
from fractions import Fraction

import numpy as np

# Випадковий індекс узгодженості RI (таблиця Сааті), ключ - розмір матриці n
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def parse_value(x):
    """Перетворює запис на кшталт '1/3' або 3 у число."""
    return float(Fraction(str(x)))


def build_matrix(upper):
    """
    Будує повну обернено-симетричну матрицю з верхнього трикутника.

    upper - список рядків, де в рядку i записані оцінки a[i][j] для j > i.
    Діагональ = 1, нижній трикутник заповнюється як a[j][i] = 1 / a[i][j].
    """
    n = len(upper) + 1
    a = np.ones((n, n))
    for i, row in enumerate(upper):
        for k, val in enumerate(row):
            j = i + 1 + k
            a[i, j] = parse_value(val)
            a[j, i] = 1.0 / a[i, j]
    return a


def check_matrix(a):
    """Перевіряє, що матриця квадратна, додатна і обернено-симетрична."""
    a = np.asarray(a, dtype=float)
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матриця має бути квадратною")
    if np.any(a <= 0):
        raise ValueError("Усі елементи матриці мають бути додатними")
    if not np.allclose(a * a.T, 1.0):
        raise ValueError("Матриця не є обернено-симетричною (a_ji != 1/a_ij)")
    return a


def priority_vector(a):
    """
    Рахує вектор пріоритетів методом власного вектора.

    Повертає (w, lambda_max): w - нормований власний вектор, що відповідає
    найбільшому власному значенню, сума його елементів = 1.
    """
    a = check_matrix(a)
    eigvals, eigvecs = np.linalg.eig(a)
    k = np.argmax(eigvals.real)
    lambda_max = eigvals[k].real
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()
    return w, lambda_max


def consistency(lambda_max, n):
    """Повертає (CI, CR). Для n <= 2 матриця завжди узгоджена."""
    if n <= 2:
        return 0.0, 0.0
    ci = (lambda_max - n) / (n - 1)
    cr = ci / RI_TABLE[n]
    return ci, cr


def analyze(a):
    """Повний аналіз однієї матриці: ваги, lambda_max, CI, CR."""
    w, lam = priority_vector(a)
    ci, cr = consistency(lam, len(w))
    return {"w": w, "lambda_max": lam, "CI": ci, "CR": cr, "ok": cr <= 0.10}


def synthesize(criteria_w, local_matrix):
    """
    Синтез глобальних пріоритетів.

    local_matrix[j][i] - локальний пріоритет i-ї альтернативи за j-м критерієм.
    P(A_i) = sum_j w_j * local_matrix[j][i]
    """
    return np.asarray(criteria_w) @ np.asarray(local_matrix)


def sensitivity(criteria_w, local_matrix, j, new_weight):
    """
    Аналіз чутливості: ставимо вагу j-го критерію = new_weight, а ваги
    решти критеріїв пропорційно масштабуємо, щоб сума лишилась 1.
    Повертає нові глобальні пріоритети.
    """
    w = np.asarray(criteria_w, dtype=float).copy()
    rest = 1.0 - w[j]
    w = w * (1.0 - new_weight) / rest
    w[j] = new_weight
    return synthesize(w, local_matrix)
