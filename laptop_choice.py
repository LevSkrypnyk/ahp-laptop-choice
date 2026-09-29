"""
laptop_choice.py - задача вибору ноутбука для розробника ПЗ методом МАІ.

Тут описані критерії, альтернативи і мої експертні оцінки (матриці парних
порівнянь). Усі розрахунки робить модуль ahp.py. Програма виводить
результати в консоль і зберігає графіки в папку results/.

Запуск:  python laptop_choice.py
"""
import os

import numpy as np

import ahp

CRITERIA = ["Продуктивність", "Автономність", "Портативність", "Вартість",
            "Екран", "Надійність", "Дизайн"]
ALTERNATIVES = ["HP", "ASUS", "Lenovo", "MacBook", "Acer"]

# Матриці задаю верхнім трикутником (рядок i: оцінки a[i][j] для j > i),
# нижня частина заповнюється автоматично як 1/a[i][j].

# Критерії відносно мети
CRITERIA_MATRIX = [
    # Авт  Порт  Варт  Екр  Над  Диз
    [2,    3,    2,    2,   1,   5],     # Продуктивність
    [      2,    1,    1,   1,   3],     # Автономність
    [            "1/2", 1, "1/2", 2],    # Портативність
    [                  1,   1,   3],     # Вартість
    [                       "1/2", 2],   # Екран
    [                              4],   # Надійність
]

# Альтернативи за кожним критерієм. Порядок: HP, ASUS, Lenovo, MacBook, Acer
ALT_MATRICES = {
    "Продуктивність": [
        ["1/2", 1, "1/2", 1],
        [1, "1/2", 2],
        ["1/2", 2],
        [3],
    ],
    "Автономність": [
        [1, 1, "1/2", 1],
        [1, "1/2", 2],
        ["1/2", 1],
        [3],
    ],
    "Портативність": [
        [1, 1, 1, 1],
        [1, 2, 1],
        [2, 1],
        ["1/2"],
    ],
    "Вартість": [
        [1, 1, 2, 1],
        [1, 2, 1],
        [1, "1/2"],
        ["1/3"],
    ],
    "Екран": [
        [1, 1, 1, 2],
        [1, 1, 1],
        [1, 1],
        [2],
    ],
    "Надійність": [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 2],
        [2],
    ],
    "Дизайн": [
        [1, 1, 1, 2],
        [1, 1, 1],
        ["1/2", 1],
        [2],
    ],
}

OUT_DIR = "results"


def print_matrix(title, names, a):
    print(f"\n{title}")
    print(" " * 16 + "".join(f"{n[:9]:>10}" for n in names))
    for name, row in zip(names, a):
        print(f"{name:<16}" + "".join(f"{v:>10.3f}" for v in row))


def print_result(res, names):
    for i in np.argsort(-res["w"]):
        print(f"  {names[i]:<16}{res['w'][i]:.4f}")
    status = "узгоджено" if res["ok"] else "НЕУЗГОДЖЕНО, треба переглянути оцінки"
    print(f"  lambda_max = {res['lambda_max']:.4f}, CI = {res['CI']:.4f}, "
          f"CR = {res['CR']:.4f} ({status})")


def leader_changes(crit_w, local, j, step=0.001):
    """
    Шукає, при яких вагах j-го критерію змінюється лідер рейтингу.
    Вагу перебираємо до 0.99: при вазі рівно 1 решта критеріїв зникає
    і часто виходять однакові пріоритети, це не цікаво.
    """
    changes = []
    prev = None
    for x in np.arange(0, 0.99 + step / 2, step):
        g = ahp.sensitivity(crit_w, local, j, x)
        leader = ALTERNATIVES[int(np.argmax(g))]
        if leader != prev:
            changes.append((round(float(x), 3), leader))
            prev = leader
    return changes


def make_charts(crit_w, local, glob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) ваги критеріїв
    order = np.argsort(-crit_w)
    plt.figure(figsize=(8, 4.5))
    plt.bar([CRITERIA[i] for i in order], crit_w[order], color="#4472C4")
    for k, i in enumerate(order):
        plt.text(k, crit_w[i] + 0.004, f"{crit_w[i]:.3f}", ha="center")
    plt.ylabel("Вага критерію")
    plt.title("Вагові коефіцієнти критеріїв")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "criteria_weights.png"), dpi=200)
    plt.close()

    # 2) підсумковий рейтинг
    order = np.argsort(-glob)
    plt.figure(figsize=(8, 4.5))
    colors = ["#2E7D32" if k == 0 else "#90A4AE" for k in range(len(order))]
    plt.bar([ALTERNATIVES[i] for i in order], glob[order], color=colors)
    for k, i in enumerate(order):
        plt.text(k, glob[i] + 0.003, f"{glob[i]:.4f}", ha="center")
    plt.ylabel("Глобальний пріоритет")
    plt.title("Інтегральні пріоритети альтернатив")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "global_priorities.png"), dpi=200)
    plt.close()

    # 3) внесок кожного критерію в глобальний пріоритет альтернативи
    contrib = crit_w[:, None] * local          # [критерій, альтернатива]
    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(ALTERNATIVES))
    for j, name in enumerate(CRITERIA):
        plt.bar(ALTERNATIVES, contrib[j], bottom=bottom, label=name)
        bottom += contrib[j]
    plt.ylabel("Внесок у глобальний пріоритет")
    plt.title("Структура глобальних пріоритетів за критеріями")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "contribution.png"), dpi=200)
    plt.close()

    # 4) чутливість рейтингу до ваги критерію "Вартість"
    j = CRITERIA.index("Вартість")
    xs = np.linspace(0, 0.8, 161)
    ys = np.array([ahp.sensitivity(crit_w, local, j, x) for x in xs])
    plt.figure(figsize=(8, 4.5))
    for i, name in enumerate(ALTERNATIVES):
        plt.plot(xs, ys[:, i], label=name, linewidth=2)
    plt.axvline(crit_w[j], color="gray", linestyle="--")
    plt.text(crit_w[j] + 0.01, ys.max(), "поточна вага", color="gray", va="top")
    plt.xlabel("Вага критерію «Вартість»")
    plt.ylabel("Глобальний пріоритет")
    plt.title("Чутливість рейтингу до ваги критерію «Вартість»")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sensitivity_cost.png"), dpi=200)
    plt.close()


def main():
    # крок 1: ваги критеріїв
    cm = ahp.build_matrix(CRITERIA_MATRIX)
    print_matrix("Матриця парних порівнянь критеріїв", CRITERIA, cm)
    crit = ahp.analyze(cm)
    print("Ваги критеріїв:")
    print_result(crit, CRITERIA)

    # крок 2: локальні пріоритети альтернатив за кожним критерієм
    local = []
    all_cr = [crit["CR"]]
    for name in CRITERIA:
        m = ahp.build_matrix(ALT_MATRICES[name])
        res = ahp.analyze(m)
        print_matrix(f"Критерій: {name}", ALTERNATIVES, m)
        print_result(res, ALTERNATIVES)
        local.append(res["w"])
        all_cr.append(res["CR"])
    local = np.array(local)

    # крок 3: синтез глобальних пріоритетів
    glob = ahp.synthesize(crit["w"], local)
    print("\nГлобальні пріоритети (підсумковий рейтинг):")
    for place, i in enumerate(np.argsort(-glob), start=1):
        print(f"  {place}. {ALTERNATIVES[i]:<10}{glob[i]:.4f}")
    print(f"\nМаксимальний CR серед усіх матриць: {max(all_cr):.4f}")
    print(f"Оптимальний вибір: {ALTERNATIVES[int(np.argmax(glob))]}")

    # крок 4: аналіз чутливості (при якій вазі критерію змінюється лідер)
    print("\nАналіз чутливості (вага критерію -> новий лідер):")
    for j, name in enumerate(CRITERIA):
        changes = leader_changes(crit["w"], local, j)
        text = ", ".join(f"з {x:.3f} - {who}" for x, who in changes[1:])
        print(f"  {name:<16}(зараз {crit['w'][j]:.3f}): "
              f"{text if text else 'лідер не змінюється'}")

    try:
        make_charts(crit["w"], local, glob)
        print(f"\nГрафіки збережено в папку {OUT_DIR}/")
    except ImportError:
        print("\nmatplotlib не встановлено - графіки не побудовано")


if __name__ == "__main__":
    main()
