import sqlite3
import pandas as pd
from datetime import datetime
import threading

class Database:
    def __init__(self, db_name='currency_data.db'):
        """Инициализация подключения к базе данных SQLite"""
        self.db_name = db_name
        self.thread_local = threading.local()
        self.create_tables()
    
    def get_connection(self):
        """Получение соединения для текущего потока"""
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(self.db_name)
        return self.thread_local.conn
    
    def get_cursor(self):
        """Получение курсора для текущего потока"""
        return self.get_connection().cursor()
    
    def create_tables(self):
        """Создание необходимых таблиц, если они не существуют"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Таблица для хранения курсов валют
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS currency_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            currency_code TEXT NOT NULL,
            nominal INTEGER NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Таблица для хранения прогнозов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            currency_code TEXT NOT NULL,
            date_predicted TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            model_version TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
    
    def save_currency_data(self, data):
        """Сохранение данных о валютах в базу данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for currency in data:
            # Проверка, существует ли уже такая запись
            cursor.execute('''
            SELECT id FROM currency_rates 
            WHERE date=? AND currency_code=?
            ''', (currency['date'], currency['currency_code']))
            
            existing = cursor.fetchone()
            
            if existing:
                # Обновление существующей записи
                cursor.execute('''
                UPDATE currency_rates 
                SET nominal=?, name=?, value=?, timestamp=CURRENT_TIMESTAMP
                WHERE date=? AND currency_code=?
                ''', (
                    currency['nominal'], 
                    currency['name'], 
                    currency['value'], 
                    currency['date'], 
                    currency['currency_code']
                ))
            else:
                # Добавление новой записи
                cursor.execute('''
                INSERT INTO currency_rates (date, currency_code, nominal, name, value)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    currency['date'],
                    currency['currency_code'],
                    currency['nominal'],
                    currency['name'],
                    currency['value']
                ))
        
        conn.commit()
    
    def save_prediction(self, currency_code, date_predicted, predicted_value, model_version):
        """Сохранение прогноза в базу данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO predictions (currency_code, date_predicted, predicted_value, model_version)
        VALUES (?, ?, ?, ?)
        ''', (currency_code, date_predicted, predicted_value, model_version))
        
        conn.commit()
    
    def get_currency_history(self, currency_code, days=30):
        """Получение истории курса валюты за указанное количество дней"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT date, value, nominal FROM currency_rates
        WHERE currency_code=?
        ORDER BY date DESC
        LIMIT ?
        ''', (currency_code, days))
        
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows, columns=['date', 'value', 'nominal'])
        # Преобразуем значение к номиналу 1
        df['value'] = df['value'] / df['nominal']
        # Переворачиваем DataFrame для хронологического порядка
        df = df.iloc[::-1].reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_all_currencies(self):
        """Получение списка всех доступных валют"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT DISTINCT currency_code, name FROM currency_rates
        ORDER BY name
        ''')
        
        return cursor.fetchall()
    
    def get_last_update_date(self):
        """Получение даты последнего обновления данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT MAX(date) FROM currency_rates
        ''')
        
        result = cursor.fetchone()
        return result[0] if result and result[0] else None
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()
            del self.thread_local.conn