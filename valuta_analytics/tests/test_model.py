import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.model import ImprovedCurrencyPredictor


class TestImprovedCurrencyPredictor(unittest.TestCase):
    def setUp(self):
        # Создаем временную директорию для моделей
        self.test_dir = tempfile.mkdtemp()
        self.predictor = ImprovedCurrencyPredictor(model_dir=self.test_dir)
        
        # Генерируем тестовые данные
        self.dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        self.data = pd.DataFrame({
            'date': self.dates,
            'value': values
        })

    def tearDown(self):
        # Удаляем временную директорию
        shutil.rmtree(self.test_dir)

    def test_prepare_features(self):
        """Тестируем подготовку признаков"""
        X, y, features = self.predictor._prepare_features(self.data, engineer_features=True)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        self.assertEqual(len(X), len(y))

    def test_differencing(self):
        """Тестируем дифференцирование данных"""
        diff_data, is_differenced = self.predictor.apply_differencing(self.data)
        self.assertTrue(is_differenced)
        self.assertEqual(len(diff_data), len(self.data) - 1)
        self.assertIn('value_diff', diff_data.columns)
        self.assertIn('value_original', diff_data.columns)

    def test_outlier_detection(self):
        """Тестируем обнаружение выбросов"""
        # Добавляем явный выброс
        data = self.data.copy()
        data.loc[50, 'value'] += 10  # создаем искусственный выброс
        
        outliers_zscore, mask_zscore = self.predictor.identify_outliers(data, method='zscore')
        outliers_iqr, mask_iqr = self.predictor.identify_outliers(data, method='iqr')

        self.assertGreater(len(outliers_zscore), 0)
        self.assertGreater(len(outliers_iqr), 0)
        self.assertTrue(mask_zscore.iloc[50])
        self.assertTrue(mask_iqr.iloc[50])

    def test_outlier_handling(self):
        """Тестируем обработку выбросов"""
        data = self.data.copy()
        data.loc[50, 'value'] += 10  # создаем искусственный выброс

        # Winsorize
        cleaned_winsor = self.predictor.handle_outliers(data, method='winsorize')
        self.assertAlmostEqual(cleaned_winsor.loc[50, 'value'], data['value'].quantile(0.95), delta=0.5)

        # Mean
        cleaned_mean = self.predictor.handle_outliers(data, method='mean')
        self.assertAlmostEqual(cleaned_mean.loc[50, 'value'], data['value'].mean(), delta=0.5)

        # Median
        cleaned_median = self.predictor.handle_outliers(data, method='median')
        self.assertAlmostEqual(cleaned_median.loc[50, 'value'], data['value'].median(), delta=0.5)

        # Remove
        cleaned_remove = self.predictor.handle_outliers(data, method='remove')
        self.assertLess(len(cleaned_remove), len(data))

    def test_train_and_predict(self):
        """Тестируем обучение и прогнозирование"""
        success = self.predictor.train(self.data, optimize=False, feature_engineering=True, handle_outliers_method='winsorize')
        self.assertTrue(success)

        predictions = self.predictor.predict_next_days(self.data, days=7)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 7)
        self.assertIn('predicted_value', predictions.columns)

    def test_model_saving_loading(self):
        """Тестируем сохранение и загрузку модели"""
        # Обучаем модель
        success = self.predictor.train(self.data, optimize=False, feature_engineering=True)
        self.assertTrue(success)

        # Сохраняем версию
        version = self.predictor.version

        # Создаем новый экземпляр и загружаем модель
        new_predictor = ImprovedCurrencyPredictor(model_dir=self.test_dir)
        loaded = new_predictor.load_model(version=version)

        self.assertTrue(loaded)
        self.assertEqual(new_predictor.version, version)
        self.assertEqual(new_predictor.best_model_type, self.predictor.best_model_type)

        # Пробуем сделать прогноз
        predictions = new_predictor.predict_next_days(self.data, days=7)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 7)

    def test_metrics(self):
        """Тестируем расчет метрик качества модели"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        metrics = self.predictor._calculate_metrics(y_true, y_pred)

        self.assertIn('r2', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertGreater(metrics['r2'], 0.9)


if __name__ == '__main__':
    unittest.main()