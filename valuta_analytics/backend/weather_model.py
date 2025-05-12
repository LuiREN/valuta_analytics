import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import joblib
import os

class WeatherPredictor:
    def __init__(self, model_dir='models/weather'):
        """Инициализация предиктора для прогнозирования погоды"""
        self.model_dir = model_dir
        # Создаем директорию для моделей, если она не существует
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model = None
        self.scaler = StandardScaler()
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}  # Словарь для хранения метрик модели
        self.city = None  # Город, для которого обучена модель
    
    def prepare_features(self, df):
        """Подготовка признаков для модели"""
        # Преобразуем даты в datetime, если они еще не в этом формате
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Создаем признаки, связанные с датой
        df_features = pd.DataFrame()
        
        # День года (синусоида для учета цикличности)
        day_of_year = df['date'].dt.dayofyear
        df_features['sin_day'] = np.sin(2 * np.pi * day_of_year / 365)
        df_features['cos_day'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Месяц (синусоида для учета цикличности)
        month = df['date'].dt.month
        df_features['sin_month'] = np.sin(2 * np.pi * month / 12)
        df_features['cos_month'] = np.cos(2 * np.pi * month / 12)
        
        # День недели (синусоида для учета цикличности)
        day_of_week = df['date'].dt.dayofweek
        df_features['sin_week'] = np.sin(2 * np.pi * day_of_week / 7)
        df_features['cos_week'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Добавляем тренд
        df_features['trend'] = (df['date'] - df['date'].min()).dt.days
        
        # Если есть, добавляем дополнительные признаки
        if 'humidity' in df.columns:
            df_features['humidity'] = df['humidity']
        if 'pressure' in df.columns:
            df_features['pressure'] = df['pressure']
        if 'wind_speed' in df.columns:
            df_features['wind_speed'] = df['wind_speed']
        
        # Добавляем квадратичные признаки для улучшения обучения
        if 'humidity' in df.columns:
            df_features['humidity_squared'] = df['humidity'] ** 2
        if 'pressure' in df.columns:
            df_features['pressure_squared'] = df['pressure'] ** 2
        
        # Добавляем взаимодействия между признаками
        if 'humidity' in df.columns and 'pressure' in df.columns:
            df_features['humidity_pressure'] = df['humidity'] * df['pressure']
            
        # Добавляем сезонные признаки
        df_features['is_winter'] = ((month >= 12) | (month <= 2)).astype(int)
        df_features['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)
        df_features['is_summer'] = ((month >= 6) & (month <= 8)).astype(int)
        df_features['is_autumn'] = ((month >= 9) & (month <= 11)).astype(int)
        
        # Добавляем лаги температуры, если есть достаточно данных
        if len(df) > 7:
            for lag in range(1, min(8, len(df))):
                df_features[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        
        # Добавляем скользящие средние, если есть достаточно данных
        if len(df) > 7:
            df_features['temp_ma3'] = df['temperature'].rolling(window=3).mean()
            if len(df) > 10:
                df_features['temp_ma7'] = df['temperature'].rolling(window=7).mean()
        
        # Добавляем амплитуду температур за последние дни
        if len(df) > 7:
            df_features['temp_amp3'] = df['temperature'].rolling(window=3).max() - df['temperature'].rolling(window=3).min()
            if len(df) > 10:
                df_features['temp_amp7'] = df['temperature'].rolling(window=7).max() - df['temperature'].rolling(window=7).min()
        
        # Удаляем строки с NaN значениями
        df_features = df_features.dropna()
        
        # Целевая переменная
        y = df.loc[df_features.index, 'temperature'].values
        
        return df_features.values, y
    
    def train(self, df, model_type='gradient_boosting', city=None):
        """Обучение модели на исторических данных"""
        if df.empty:
            print("Нет данных для обучения модели")
            return False
        
        # Сохраняем название города
        self.city = city
        
        # Если данных слишком мало, добавляем синтетические данные
        if len(df) < 30:
            print(f"Внимание: мало данных для обучения ({len(df)}). Добавляем синтетические данные.")
            # Генерируем синтетические данные на основе существующих
            synthetic_data = []
            base_date = pd.to_datetime(df['date'].min()) - timedelta(days=30)
            
            for i in range(30):
                date = base_date + timedelta(days=i)
                # Используем среднюю температуру с небольшим шумом
                mean_temp = df['temperature'].mean()
                temp = mean_temp + np.random.normal(0, 2)
                
                # Добавляем синтетические данные с остальными параметрами
                row = {
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature': temp
                }
                
                # Копируем другие параметры (если есть)
                if 'humidity' in df.columns:
                    row['humidity'] = df['humidity'].mean() + np.random.normal(0, 5)
                if 'pressure' in df.columns:
                    row['pressure'] = df['pressure'].mean() + np.random.normal(0, 3)
                if 'wind_speed' in df.columns:
                    row['wind_speed'] = df['wind_speed'].mean() + np.random.normal(0, 1)
                
                synthetic_data.append(row)
            
            # Добавляем синтетические данные к исходным
            synthetic_df = pd.DataFrame(synthetic_data)
            df = pd.concat([synthetic_df, df], ignore_index=True)
            
            # Сортируем по дате
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # Разделяем данные на обучающую и тестовую выборки (80/20)
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size].copy()
        df_test = df.iloc[train_size:].copy()
        
        # Подготовка признаков
        X_train, y_train = self.prepare_features(df_train)
        
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Выбираем тип модели
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42,
                subsample=0.8,
                min_samples_split=3
            )
        else:  # По умолчанию Random Forest
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=2,
                random_state=42
            )
        
        # Обучение модели
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка на тестовой выборке
        if len(df_test) > 5:
            X_test, y_test = self.prepare_features(df_test)
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            # Рассчитываем метрики
            self.metrics = self.calculate_metrics(y_test, y_pred)
        else:
            # Если недостаточно данных, оцениваем на обучающей выборке
            y_pred = self.model.predict(X_train_scaled)
            self.metrics = self.calculate_metrics(y_train, y_pred)
            print("Внимание: недостаточно данных для валидации. Использованы обучающие данные.")
        
        # Сохранение модели
        self.save_model()
        print(f"Модель {model_type} обучена. Метрики: {self.metrics}")
        
        return True
    
    def calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества модели"""
        metrics = {}
        
        # R-квадрат (коэффициент детерминации)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Средняя абсолютная ошибка (MAE)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Корень из среднеквадратичной ошибки (RMSE)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Средняя процентная ошибка (MAPE)
        # Избегаем деления на ноль
        mask = y_true != 0
        if any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = 0
        
        return metrics
    
    def predict(self, dates, additional_features=None):
        """Прогнозирование температуры на указанные даты"""
        if self.model is None:
            print("Модель не обучена")
            return None
        
        # Преобразуем даты в DataFrame
        df = pd.DataFrame({'date': dates})
        
        # Добавляем дополнительные признаки, если они есть
        if additional_features is not None:
            for key, values in additional_features.items():
                df[key] = values
        
        # Добавляем пустой столбец для температуры (необходим для вызова prepare_features)
        df['temperature'] = np.nan
        
        # Создаем признаки для дат без учета лагов
        df_features = pd.DataFrame()
        
        # День года (синусоида для учета цикличности)
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear
        df_features['sin_day'] = np.sin(2 * np.pi * day_of_year / 365)
        df_features['cos_day'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Месяц (синусоида для учета цикличности)
        month = pd.to_datetime(df['date']).dt.month
        df_features['sin_month'] = np.sin(2 * np.pi * month / 12)
        df_features['cos_month'] = np.cos(2 * np.pi * month / 12)
        
        # День недели (синусоида для учета цикличности)
        day_of_week = pd.to_datetime(df['date']).dt.dayofweek
        df_features['sin_week'] = np.sin(2 * np.pi * day_of_week / 7)
        df_features['cos_week'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Добавляем тренд (относительно текущей даты)
        min_date = pd.to_datetime(pd.to_datetime(df['date']).min())
        df_features['trend'] = (pd.to_datetime(df['date']) - min_date).dt.days
        
        # Добавляем сезонные признаки
        df_features['is_winter'] = ((month >= 12) | (month <= 2)).astype(int)
        df_features['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)
        df_features['is_summer'] = ((month >= 6) & (month <= 8)).astype(int)
        df_features['is_autumn'] = ((month >= 9) & (month <= 11)).astype(int)
        
        # Добавляем дополнительные признаки, если они есть
        if additional_features is not None:
            for key in additional_features.keys():
                if key in ['humidity', 'pressure', 'wind_speed']:
                    df_features[key] = df[key]
                    
                    # Добавляем квадратичные признаки
                    if key == 'humidity':
                        df_features['humidity_squared'] = df['humidity'] ** 2
                    if key == 'pressure':
                        df_features['pressure_squared'] = df['pressure'] ** 2
            
            # Добавляем взаимодействия между признаками, если есть
            if 'humidity' in additional_features and 'pressure' in additional_features:
                df_features['humidity_pressure'] = df['humidity'] * df['pressure']
        
        # Преобразуем признаки
        X = df_features.values
        
        # Проверяем соответствие признаков
        expected_features = self.model.n_features_in_
        if X.shape[1] != expected_features:
            print(f"Внимание: количество признаков не соответствует модели. Ожидается {expected_features}, получено {X.shape[1]}")
            # Заполняем остальные признаки нулями
            X_padded = np.zeros((X.shape[0], expected_features))
            X_padded[:, :X.shape[1]] = X
            X = X_padded
        
        X_scaled = self.scaler.transform(X)
        
        # Делаем прогноз
        predictions = self.model.predict(X_scaled)
        
        # Формируем результат
        result = []
        for i, date in enumerate(dates):
            result.append({
                'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date,
                'predicted_temperature': float(predictions[i]),
                'confidence': float(self.metrics.get('r2', 0.5))  # Используем R² как показатель уверенности
            })
        
        return result
    
    def save_model(self):
        """Сохранение модели и метаданных"""
        # Создаем директорию для города, если её нет
        city_dir = os.path.join(self.model_dir, self.city) if self.city else self.model_dir
        if not os.path.exists(city_dir):
            os.makedirs(city_dir)
        
        # Пути для сохранения
        model_path = os.path.join(city_dir, f"model_{self.version}.joblib")
        scaler_path = os.path.join(city_dir, f"scaler_{self.version}.joblib")
        metrics_path = os.path.join(city_dir, f"metrics_{self.version}.joblib")
        metadata_path = os.path.join(city_dir, f"metadata_{self.version}.joblib")
        
        # Сохраняем модель и связанные данные
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.metrics, metrics_path)
        
        # Сохраняем метаданные
        metadata = {
            'version': self.version,
            'city': self.city,
            'model_type': type(self.model).__name__,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(metadata, metadata_path)
        
        print(f"Модель сохранена: {model_path}")
    
    def load_model(self, city, version=None):
        """Загрузка модели по городу и опционально версии"""
        # Директория для города
        city_dir = os.path.join(self.model_dir, city)
        if not os.path.exists(city_dir):
            print(f"Нет моделей для города {city}")
            return False
        
        # Если версия не указана, ищем последнюю
        if version is None:
            model_files = [f for f in os.listdir(city_dir) if f.startswith("model_")]
            if not model_files:
                print(f"Нет сохраненных моделей для города {city}")
                return False
            
            model_files.sort(reverse=True)
            version = model_files[0].replace("model_", "").replace(".joblib", "")
        
        # Пути к файлам модели
        model_path = os.path.join(city_dir, f"model_{version}.joblib")
        scaler_path = os.path.join(city_dir, f"scaler_{version}.joblib")
        metrics_path = os.path.join(city_dir, f"metrics_{version}.joblib")
        
        # Проверяем наличие файлов
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Файлы модели версии {version} для города {city} не найдены")
            return False
        
        # Загружаем модель и скейлер
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.version = version
        self.city = city
        
        # Загружаем метрики, если они есть
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {}
        
        print(f"Загружена модель для города {city}, версия {version}")
        return True
    
    def get_model_metrics(self):
        """Получение метрик модели"""
        return self.metrics
    
    def get_model_info(self):
        """Получение информации о модели"""
        return {
            'version': self.version,
            'city': self.city,
            'model_type': type(self.model).__name__ if self.model else None,
            'metrics': self.metrics
        }