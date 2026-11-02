"""Автотесты программы SimpleAnalysis: python -m unittest discover tests -v"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_analysis as sa  # noqa: E402


class TestGeneration(unittest.TestCase):
    """Модуль получения данных."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'dataset.csv')

    def tearDown(self):
        self.tmp.cleanup()

    def test_size_and_range(self):
        sa.generate_dataset(self.path)
        series, _ = sa.load_and_clean(self.path)
        self.assertEqual(len(series), sa.SIZE)
        self.assertTrue(series.between(sa.LOW, sa.HIGH).all())

    def test_reproducible(self):
        sa.generate_dataset(self.path)
        first, _ = sa.load_and_clean(self.path)
        sa.generate_dataset(self.path)
        second, _ = sa.load_and_clean(self.path)
        self.assertTrue(first.equals(second))  # зерно = ID студента


class TestCleaning(unittest.TestCase):
    """Модуль очистки данных от цифрового мусора."""

    def clean(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'd.csv')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('value\n' + '\n'.join(lines) + '\n')
            return sa.load_and_clean(path)

    def test_garbage_removed(self):
        series, report = self.clean(['5', 'abc', '', '12.5', '15000', '-20001', 'nan', '-10000', '10000'])
        self.assertEqual(series.tolist(), [5, -10000, 10000])
        self.assertEqual(report['дробных'], 1)
        self.assertEqual(report['вне диапазона'], 2)
        self.assertEqual(report['осталось после очистки'], 3)

    def test_spaces_and_integer_float(self):
        series, _ = self.clean([' 7 ', '3.0'])
        self.assertEqual(series.tolist(), [7, 3])  # 3.0 - целое число


class TestStatistics(unittest.TestCase):
    """Модуль расчёта числовых характеристик."""

    def test_known_values(self):
        s = pd.Series([1, 2, 2, 3, 3, 3])
        stats = sa.calc_statistics(s)
        self.assertEqual(stats['Минимальное значение'], 1)
        self.assertEqual(stats['Максимальное значение'], 3)
        self.assertEqual(stats['Сумма чисел'], 14)
        self.assertEqual(stats['Количество повторяющихся значений (различных чисел)'], 2)
        self.assertEqual(stats['Количество элементов с повторяющимися значениями'], 5)

    def test_std_manual_matches_pandas(self):
        values = list(np.random.default_rng(1).integers(-100, 100, 50))
        self.assertAlmostEqual(sa.std_manual(values), pd.Series(values).std(), places=9)

    def test_round_half_up(self):
        cases = {250: 300, -250: -300, 149: 100, 150: 200, 5050: 5100, -9951: -10000, 0: 0}
        for value, expected in cases.items():
            self.assertEqual(sa.round_half_up(value), expected, value)
        self.assertEqual(round(250, -2), 200)  # банковское округление round() - причина своей функции


class TestProcessing(unittest.TestCase):
    """Модули промежуточного анализа."""

    def setUp(self):
        self.s = pd.Series([-10000, -1, 0, 1, 2499, 2500, 2501, 10000])

    def test_split_by_sign(self):
        parts = sa.split_by_sign(self.s)
        self.assertEqual(len(parts['отрицательные']), 2)
        self.assertEqual(len(parts['нулевые']), 1)
        self.assertEqual(len(parts['положительные']), 5)

    def test_sums_by_ranges_total(self):
        table = sa.sums_by_ranges(self.s)
        self.assertEqual(table['count'].sum(), len(self.s))
        self.assertEqual(table['sum'].sum(), self.s.sum())

    def test_build_frame(self):
        frame = sa.build_frame(self.s)
        self.assertEqual(list(frame.columns), ['value', 'sorted_asc', 'sorted_desc', 'rounded_100'])
        self.assertTrue(frame['sorted_asc'].is_monotonic_increasing)
        self.assertTrue(frame['sorted_desc'].is_monotonic_decreasing)


class TestProject(unittest.TestCase):
    """Комплексный тест: полный запуск программы."""

    def test_main_creates_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = sa.OUT_DIR
            sa.OUT_DIR = tmp
            try:
                with open(os.devnull, 'w') as null:
                    stdout, sys.stdout = sys.stdout, null
                    try:
                        sa.main()
                    finally:
                        sys.stdout = stdout
            finally:
                sa.OUT_DIR = old
            for name in ('dataset.csv', 'statistics.txt', '01_line.png', '02_histogram.png',
                         '03_sorted.png', '04_merged.png', 'dataframe.csv', 'merged.csv'):
                self.assertTrue(os.path.getsize(os.path.join(tmp, name)) > 0, name)
            merged = pd.read_csv(os.path.join(tmp, 'merged.csv'))
            self.assertEqual(merged.shape, (1000, 2))
            self.assertFalse(math.isnan(merged['uniform'].corr(merged['normal'])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
