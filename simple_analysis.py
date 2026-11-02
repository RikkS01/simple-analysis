"""Индивидуальное задание «SimpleAnalysis» (учебная ознакомительная практика).

Программа:
1) генерирует набор из 1000 целых чисел в диапазоне [-10000; 10000]
   и сохраняет его в файл dataset.csv;
2) загружает набор из файла, очищает от «цифрового мусора» и формирует Series;
3) рассчитывает стандартные числовые характеристики и выводит их
   в консоль и в файл statistics.txt;
4) строит линейный график и гистограмму значений, округлённых до сотен;
5) формирует DataFrame со столбцами отсортированных значений
   и строит графики отсортированных рядов;
6) разделяет данные по признакам, считает суммы по диапазонам,
   загружает второй набор из другого источника и объединяет наборы.

Автор: Яцко Сорин, студенческий ID 70204383.
"""

import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # сохранение графиков в файлы без окна
import matplotlib.pyplot as plt  # noqa: E402

STUDENT_ID = 70204383
SIZE = 1000
LOW, HIGH = -10000, 10000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# ---------------------------------------------------------------------------
# 1. Получение данных
# ---------------------------------------------------------------------------
def generate_dataset(path, size=SIZE, seed=STUDENT_ID):
    """Сгенерировать целые числа и сохранить их в csv-файл (столбец value)."""
    rng = np.random.default_rng(seed)  # воспроизводимая генерация
    values = rng.integers(LOW, HIGH, size=size, endpoint=True)
    pd.DataFrame({'value': values}).to_csv(path, index=False)
    # имитация сбоев при передаче данных: «цифровой мусор» в файле
    with open(path, 'a', encoding='utf-8') as file:
        file.write('abc\n\n12.5\n15000\n-20001\nnan\n')
    return path


def load_and_clean(path):
    """Загрузить набор из csv и удалить «цифровой мусор».

    Мусором считаются пустые строки, нечисловые значения, дробные числа
    и значения вне диапазона [-10000; 10000].
    """
    raw = pd.read_csv(path, dtype=str)['value']
    numbers = pd.to_numeric(raw.str.strip(), errors='coerce')  # текст -> NaN
    mask = numbers.notna() & (numbers % 1 == 0) & numbers.between(LOW, HIGH)
    report = {
        'всего строк': len(raw),
        'пустых и нечисловых': int(numbers.isna().sum()),
        'дробных': int((numbers.notna() & (numbers % 1 != 0)).sum()),
        'вне диапазона': int((numbers.notna() & ~numbers.between(LOW, HIGH)).sum()),
    }
    series = numbers[mask].astype(int).reset_index(drop=True)
    series.name = 'value'
    report['осталось после очистки'] = len(series)
    return series, report


# ---------------------------------------------------------------------------
# 2. Числовые характеристики
# ---------------------------------------------------------------------------
def count_repeated(series):
    """Количество значений, которые встречаются в наборе более одного раза.

    Возвращает пару: (сколько различных чисел повторяется,
    сколько элементов набора приходится на повторяющиеся числа).
    """
    counts = series.value_counts()
    repeated = counts[counts > 1]
    return len(repeated), int(repeated.sum())


def calc_statistics(series):
    """Рассчитать стандартные характеристики набора."""
    distinct_repeated, items_repeated = count_repeated(series)
    return {
        'Минимальное значение': int(series.min()),
        'Максимальное значение': int(series.max()),
        'Количество повторяющихся значений (различных чисел)': distinct_repeated,
        'Количество элементов с повторяющимися значениями': items_repeated,
        'Сумма чисел': int(series.sum()),
        'Среднее арифметическое': round(float(series.mean()), 3),
        'Среднеквадратическое отклонение (выборочное, ddof=1)': round(float(series.std()), 3),
        'Среднеквадратическое отклонение (генеральное, ddof=0)': round(
            float(series.std(ddof=0)), 3
        ),
    }


def std_manual(values):
    """Контрольный расчёт СКО по формуле без использования pandas."""
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def print_and_save(stats, path):
    """Вывести характеристики в консоль с пояснениями и сохранить в файл."""
    lines = ['Стандартные числовые характеристики набора данных:']
    lines += [f'  {name}: {value}' for name, value in stats.items()]
    text = '\n'.join(lines)
    print(text)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text + '\n')


# ---------------------------------------------------------------------------
# 3. Визуализация
# ---------------------------------------------------------------------------
def round_half_up(value, base=100):
    """Округлить до сотен по математическому правилу (0,5 - от нуля).

    Встроенная round() использует «банковское» округление
    (round(250, -2) == 200), поэтому реализовано своё правило.
    """
    sign = -1 if value < 0 else 1
    return sign * int(math.floor(abs(value) / base + 0.5)) * base


def plot_line(series, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(series.index, series.values, linewidth=0.7, color='#1F77B4')
    ax.axhline(
        series.mean(), color='#D62728', linestyle='--', label=f'среднее = {series.mean():.1f}'
    )
    ax.set_title('Линейный график исходного набора данных (1000 значений)')
    ax.set_xlabel('Номер элемента')
    ax.set_ylabel('Значение')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_histogram(rounded, path):
    counts = rounded.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(counts.index, counts.values, width=90, color='#2CA02C', edgecolor='#1B5E20')
    ax.set_title('Гистограмма значений, округлённых до сотен')
    ax.set_xlabel('Значение, округлённое до сотен')
    ax.set_ylabel('Количество значений')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sorted(frame, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame['sorted_asc'], label='по возрастанию', color='#1F77B4')
    ax.plot(frame.index, frame['sorted_desc'], label='по убыванию', color='#FF7F0E')
    ax.set_title('Отсортированные значения набора данных')
    ax.set_xlabel('Позиция в отсортированном ряду')
    ax.set_ylabel('Значение')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Промежуточный анализ: DataFrame, разделение, суммы по диапазонам
# ---------------------------------------------------------------------------
def build_frame(series):
    """DataFrame: исходные значения + отсортированные по возрастанию/убыванию."""
    frame = series.to_frame('value')
    frame['sorted_asc'] = series.sort_values().to_numpy()
    frame['sorted_desc'] = series.sort_values(ascending=False).to_numpy()
    frame['rounded_100'] = series.map(round_half_up)
    return frame


def split_by_sign(series):
    """Разделить данные по признаку знака числа."""
    return {
        'отрицательные': series[series < 0],
        'нулевые': series[series == 0],
        'положительные': series[series > 0],
    }


def sums_by_ranges(series, step=2500):
    """Суммы и количества значений по диапазонам шириной step."""
    bins = list(range(LOW, HIGH + step, step))
    groups = pd.cut(series, bins=bins, include_lowest=True)
    return series.groupby(groups, observed=False).agg(['count', 'sum'])


def merge_with_second_source(series, path):
    """Получить второй набор (другой источник - текстовый файл) и объединить."""
    rng = np.random.default_rng(STUDENT_ID % 1000)
    second = np.clip(np.round(rng.normal(0, 3000, SIZE)), LOW, HIGH).astype(int)
    np.savetxt(path, second, fmt='%d')  # источник 2: *.txt
    second_series = pd.Series(np.loadtxt(path, dtype=int), name='normal')
    merged = pd.concat([series.rename('uniform'), second_series], axis=1)
    return merged


def plot_merged(merged, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(
        [merged['uniform'], merged['normal']],
        bins=20,
        label=['равномерный (набор 1)', 'нормальный (набор 2)'],
        color=['#1F77B4', '#FF7F0E'],
    )
    axes[0].set_title('Распределения объединённых наборов')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Количество')
    axes[0].legend(fontsize=8)
    axes[1].scatter(merged['uniform'], merged['normal'], s=6, alpha=0.5)
    corr = merged['uniform'].corr(merged['normal'])
    axes[1].set_title(f'Зависимость наборов (r = {corr:.3f})')
    axes[1].set_xlabel('Набор 1')
    axes[1].set_ylabel('Набор 2')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return corr


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    def out(name):
        return os.path.join(OUT_DIR, name)

    print('=== 1. Получение данных ===')
    generate_dataset(out('dataset.csv'))
    print(f'Сгенерировано {SIZE} целых чисел в диапазоне [{LOW}; {HIGH}] -> dataset.csv')
    series, report = load_and_clean(out('dataset.csv'))
    print('Результат очистки:', report)
    print('Сводная информация о Series:')
    print(series.describe().round(2).to_string())

    print('\n=== 2. Числовые характеристики ===')
    stats = calc_statistics(series)
    print_and_save(stats, out('statistics.txt'))
    print(f'Контроль СКО по формуле: {std_manual(series.tolist()):.3f}')

    print('\n=== 3. Визуализация ===')
    plot_line(series, out('01_line.png'))
    frame = build_frame(series)
    plot_histogram(frame['rounded_100'], out('02_histogram.png'))
    print(
        'Примеры округления до сотен:',
        {v: round_half_up(v) for v in (250, -250, 149, 5050, -9951)},
    )
    print('Построены: 01_line.png, 02_histogram.png')

    print('\n=== 4. DataFrame с отсортированными столбцами ===')
    print(frame.head(10).to_string())
    frame.to_csv(out('dataframe.csv'), index=False)
    plot_sorted(frame, out('03_sorted.png'))
    print('Построен: 03_sorted.png')

    print('\n=== 5. Разделение и суммы по диапазонам ===')
    for name, part in split_by_sign(series).items():
        print(f'  {name}: {len(part)} шт., сумма {int(part.sum())}')
    ranges = sums_by_ranges(series)
    print(ranges.to_string())

    print('\n=== 6. Объединение с данными из другого источника ===')
    merged = merge_with_second_source(series, out('dataset_2.txt'))
    corr = plot_merged(merged, out('04_merged.png'))
    merged.to_csv(out('merged.csv'), index=False)
    print(merged.describe().round(2).to_string())
    print(f'Коэффициент корреляции наборов: {corr:.4f}')


if __name__ == '__main__':
    main()
