import unittest
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import threading

from backend.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Создаем временную базу данных перед каждым тестом"""
        self.test_db = "test_currency_data.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = Database(db_name=self.test_db)

    def tearDown(self):
        """Закрываем соединение и удаляем временную БД после тестов"""
        try:
            self.db.close()
            # Добавляем паузу для освобождения файла
            time.sleep(0.1)
            if os.path.exists(self.test_db):
                os.remove(self.test_db)
        except Exception as e:
            print(f"Ошибка при удалении файла: {e}")

    def test_tables_created(self):
        """Проверяем, что таблицы были созданы"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('currency_rates', 'predictions')")
        tables = cursor.fetchall()
        self.assertEqual(len(tables), 2)
        conn.close()

    def test_save_and_get_currency_data(self):
        """Тестируем сохранение и чтение данных о валюте"""
        data = [{
            "date": "2024-10-01",
            "currency_code": "USD",
            "nominal": 1,
            "name": "Доллар США",
            "value": 75.5
        }]

        self.db.save_currency_data(data)

        df = self.db.get_currency_history("USD", days=30)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["value"], 75.5)

    def test_update_existing_currency_data(self):
        """Тестируем обновление существующей записи"""
        old_value = {
            "date": "2024-10-01",
            "currency_code": "EUR",
            "nominal": 1,
            "name": "Евро",
            "value": 85.0
        }
        new_value = {
            "date": "2024-10-01",
            "currency_code": "EUR",
            "nominal": 1,
            "name": "Евро",
            "value": 86.0
        }

        self.db.save_currency_data([old_value])
        self.db.save_currency_data([new_value])

        df = self.db.get_currency_history("EUR", days=30)
        self.assertEqual(df.iloc[0]["value"], 86.0)

    def test_save_prediction(self):
        """Тестируем сохранение прогноза"""
        currency_code = "USD"
        date_predicted = "2024-10-05"
        predicted_value = 76.0
        model_version = "v1.0"

        self.db.save_prediction(currency_code, date_predicted, predicted_value, model_version)

        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE currency_code=?", (currency_code,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], currency_code)
        self.assertEqual(row[2], date_predicted)
        self.assertEqual(row[3], predicted_value)
        self.assertEqual(row[4], model_version)

    def test_get_all_currencies(self):
        """Тестируем получение списка всех валют"""
        data = [
            {"date": "2024-10-01", "currency_code": "USD", "nominal": 1, "name": "Доллар США", "value": 75.5},
            {"date": "2024-10-01", "currency_code": "EUR", "nominal": 1, "name": "Евро", "value": 85.0}
        ]
        self.db.save_currency_data(data)

        currencies = self.db.get_all_currencies()
        self.assertEqual(len(currencies), 2)
        codes = [code for code, _ in currencies]
        self.assertIn("USD", codes)
        self.assertIn("EUR", codes)

    def test_get_last_update_date(self):
        """Тестируем получение даты последнего обновления"""
        data = [{"date": "2024-10-01", "currency_code": "USD", "nominal": 1, "name": "Доллар США", "value": 75.5}]
        self.db.save_currency_data(data)

        last_update = self.db.get_last_update_date()
        self.assertEqual(last_update, "2024-10-01")

    def test_empty_data(self):
        """Тестируем поведение при отсутствии данных"""
        df = self.db.get_currency_history("USD", days=30)
        self.assertTrue(df.empty)

    def test_multiple_threads(self):
        """Тестируем работу с несколькими потоками"""
        def worker():
            data = [{
                "date": "2024-10-01",
                "currency_code": "USD",
                "nominal": 1,
                "name": "Доллар США",
                "value": 75.5
            }]
            db_local = Database(db_name=self.test_db)
            db_local.save_currency_data(data)
            db_local.close()

        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        df = self.db.get_currency_history("USD", days=30)
        self.assertGreaterEqual(len(df), 1)

    def test_duplicate_entries(self):
        """Тестируем поведение при дублировании записей"""
        entry = {
            "date": "2024-10-01",
            "currency_code": "USD",
            "nominal": 1,
            "name": "Доллар США",
            "value": 75.5
        }

        self.db.save_currency_data([entry])
        self.db.save_currency_data([entry])  # Дубликат

        df = self.db.get_currency_history("USD", days=30)
        self.assertEqual(len(df), 1)  # Должно быть только одно значение

    def test_invalid_date_format(self):
        """Тестируем сохранение данных с датой в неверном формате"""
        data = [{
            "date": "01-10-2024",  # Неверный формат
            "currency_code": "USD",
            "nominal": 1,
            "name": "Доллар США",
            "value": 75.5
        }]
    
        try:
            self.db.save_currency_data(data)
        except Exception as e:
            self.fail(f"save_currency_data() вызвала ошибку при неверной дате: {e}")
    
        # Проверяем, что данные сохранились
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT date FROM currency_rates WHERE currency_code='USD'")
        result = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "01-10-2024")

    def test_multiple_currencies(self):
        """Тестируем работу с несколькими валютами"""
        data = [
            {"date": "2024-10-01", "currency_code": "USD", "nominal": 1, "name": "Доллар США", "value": 75.5},
            {"date": "2024-10-01", "currency_code": "EUR", "nominal": 1, "name": "Евро", "value": 85.0},
            {"date": "2024-10-02", "currency_code": "USD", "nominal": 1, "name": "Доллар США", "value": 75.6},
            {"date": "2024-10-02", "currency_code": "EUR", "nominal": 1, "name": "Евро", "value": 85.1}
        ]

        self.db.save_currency_data(data)

        usd_df = self.db.get_currency_history("USD", days=30)
        eur_df = self.db.get_currency_history("EUR", days=30)

        self.assertEqual(len(usd_df), 2)
        self.assertEqual(len(eur_df), 2)
        self.assertEqual(usd_df.iloc[-1]["value"], 75.6)
        self.assertEqual(eur_df.iloc[-1]["value"], 85.1)


if __name__ == "__main__":
    unittest.main()