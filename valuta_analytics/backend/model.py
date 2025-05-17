import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

class ImprovedCurrencyPredictor:
    def __init__(self, model_dir='models'):
        """Инициализация улучшенного предиктора для прогнозирования курсов валют"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}
        self.best_model_type = None
        self.feature_names = None
        self.is_differenced = False
        self.original_mean = None
        self.data_properties = {}
        
    def check_stationarity(self, series):
        """Проверка стационарности временного ряда с помощью теста Дики-Фуллера"""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.values)
        p_value = result[1]
        print(f"P-value: {p_value}")
        return p_value <= 0.05
    
    def difference_series(self, df):
        """Применение дифференцирования для достижения стационарности"""
        if len(df) <= 5:
            return df, False
            
        # Сохраняем оригинальное среднее для восстановления
        self.original_mean = df['value'].mean()
        
        # Создаем копию для работы
        diff_df = df.copy()
        
        # Проверяем, стационарен ли ряд
        is_stationary = self.check_stationarity(diff_df['value'])
        
        if not is_stationary:
            print("Ряд не стационарен, применяем дифференцирование")
            # Сохраняем оригинальные значения
            diff_df['value_original'] = diff_df['value'].copy()
            
            # Применяем дифференцирование первого порядка
            diff_df['value_diff'] = diff_df['value'].diff()
            diff_df = diff_df.dropna().reset_index(drop=True)
            diff_df['value'] = diff_df['value_diff']
            
            # Проверяем, стал ли ряд стационарным
            if len(diff_df) > 5:
                is_stationary_now = self.check_stationarity(diff_df['value'])
                print(f"После дифференцирования ряд стационарен: {is_stationary_now}")
            
            return diff_df, True
        else:
            print("Ряд уже стационарен, дифференцирование не требуется")
            return diff_df, False
    
    def decompose_time_series(self, df):
        """Декомпозиция временного ряда на тренд, сезонность и остаток"""
        if len(df) < 14:  # Нужно минимум 2 недели данных для декомпозиции
            print("Недостаточно данных для декомпозиции")
            return None, None, None
            
        try:
            # Преобразуем 'date' в индекс
            ts = df.set_index('date')['value']
            
            # Разложение временного ряда
            decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            print("Временной ряд успешно разложен на компоненты")
            return trend, seasonal, residual
        except Exception as e:
            print(f"Ошибка при декомпозиции временного ряда: {e}")
            return None, None, None
    
    def handle_outliers(self, df, column='value', method='winsorize', threshold=3.0):
        """Обработка выбросов в данных"""
        if len(df) <= 5:
            return df
            
        clean_df = df.copy()
        
        if method == 'winsorize':
            # Ограничиваем экстремальные значения
            q05 = np.percentile(clean_df[column], 5)
            q95 = np.percentile(clean_df[column], 95)
            clean_df.loc[clean_df[column] < q05, column] = q05
            clean_df.loc[clean_df[column] > q95, column] = q95
            print(f"Выбросы обработаны методом winsorize (5-95 персентиль)")
        elif method == 'iqr':
            # Использование межквартильного размаха
            Q1 = clean_df[column].quantile(0.25)
            Q3 = clean_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            clean_df.loc[clean_df[column] < lower_bound, column] = lower_bound
            clean_df.loc[clean_df[column] > upper_bound, column] = upper_bound
            print(f"Выбросы обработаны методом IQR с порогом {threshold}")
        elif method == 'zscore':
            # Z-score метод
            z_scores = np.abs((clean_df[column] - clean_df[column].mean()) / clean_df[column].std())
            clean_df.loc[z_scores > threshold, column] = np.nan
            clean_df.fillna(clean_df[column].median(), inplace=True)
            print(f"Выбросы обработаны методом Z-score с порогом {threshold}")
            
        return clean_df
    
    def prepare_features(self, df, engineer_features=True):
        """Подготовка признаков для модели"""
        if df.empty or len(df) < 5:
            print("Недостаточно данных для подготовки признаков")
            return None, None, None
        
        # Создаем копию DataFrame для обработки
        processed_df = df.copy()
        
        # Добавляем признаки тренда и сезонности
        processed_df['days_from_start'] = (processed_df['date'] - processed_df['date'].min()).dt.days
        
        # Циклические признаки для дня недели и месяца
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day_of_week'] / 7)
        processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day_of_week'] / 7)
        
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
        processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)
        
        # Сезонные признаки (квартал вместо отдельных признаков для сезона)
        processed_df['quarter'] = processed_df['date'].dt.quarter
        
        # Лаги значений (сокращаем до 5 лагов)
        for i in range(1, min(6, len(processed_df))):
            processed_df[f'lag_{i}'] = processed_df['value'].shift(i)
        
        # Признаки изменения
        processed_df['return_1d'] = processed_df['value'].pct_change(1)
        processed_df['return_5d'] = processed_df['value'].pct_change(5) if len(processed_df) >= 6 else 0
        
        # Скользящие средние (только если данных достаточно)
        if len(processed_df) >= 8:
            processed_df['ma3'] = processed_df['value'].rolling(window=3).mean()
            processed_df['ma5'] = processed_df['value'].rolling(window=5).mean()
            processed_df['ewma'] = processed_df['value'].ewm(span=5).mean()  # Экспоненциальное скользящее среднее
            
            # Волатильность
            processed_df['volatility'] = processed_df['value'].rolling(window=5).std()
        
        # Удаляем строки с NaN значениями
        processed_df = processed_df.dropna().reset_index(drop=True)
        
        if processed_df.empty:
            print("После удаления NaN значений не осталось данных")
            return None, None, None
        
        # Выбираем признаки для модели
        if engineer_features:
            features = [
                'days_from_start', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 
                'quarter', 'return_1d'
            ]
            
            # Добавляем лаги
            lag_features = [f'lag_{i}' for i in range(1, 6) if f'lag_{i}' in processed_df.columns]
            features.extend(lag_features)
            
            # Добавляем скользящие средние и волатильность, если они есть
            if 'ma3' in processed_df.columns:
                features.extend(['ma3', 'ma5', 'ewma', 'volatility'])
                
            if 'return_5d' in processed_df.columns:
                features.append('return_5d')
        else:
            # Базовый набор признаков
            features = ['days_from_start', 'day_sin', 'day_cos']
            if 'lag_1' in processed_df.columns:
                features.append('lag_1')
        
        # Проверяем, что все выбранные признаки есть в данных
        features = [f for f in features if f in processed_df.columns]
        
        if not features:
            print("Не удалось подготовить признаки")
            return None, None, None
        
        # Формируем X и y
        X = processed_df[features].values
        y = processed_df['value'].values
        
        return X, y, features
    
    def build_arima_model(self, df, order=(5,1,0)):
        """Построение ARIMA модели"""
        if len(df) < 10:
            print("Недостаточно данных для ARIMA модели")
            return None
            
        try:
            # Создаем временной ряд
            ts = df.set_index('date')['value']
            
            # Пробуем автоматически подобрать параметры
            import pmdarima as pm
            arima_auto = pm.auto_arima(
                ts,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=1,
                seasonal=True,
                m=7,  # Недельная сезонность
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            best_order = arima_auto.order
            best_seasonal_order = arima_auto.seasonal_order
            
            print(f"Оптимальные параметры ARIMA: {best_order}, сезонные: {best_seasonal_order}")
            
            # Создаем модель с найденными параметрами
            model = SARIMAX(
                ts, 
                order=best_order, 
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False
            )
            
            results = model.fit(disp=False)
            print(f"ARIMA модель успешно обучена, AIC: {results.aic}")
            
            return results
        except Exception as e:
            print(f"Ошибка при построении ARIMA модели: {e}")
            # Пробуем построить простую ARIMA модель
            try:
                model = ARIMA(ts, order=order)
                results = model.fit()
                print(f"Базовая ARIMA модель успешно обучена, AIC: {results.aic}")
                return results
            except Exception as e2:
                print(f"Ошибка при построении базовой ARIMA модели: {e2}")
                return None
    
    def train_ml_models(self, X_train, y_train, X_test, y_test):
        """Обучение и сравнение различных ML моделей"""
        models = {
            'Ridge': Ridge(alpha=1.0),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Обучаем модель
                model.fit(X_train, y_train)
                
                # Делаем прогноз на тестовой выборке
                y_pred = model.predict(X_test)
                
                # Считаем метрики
                metrics = self._calculate_metrics(y_test, y_pred)
                results[name] = {'model': model, 'metrics': metrics}
                
                print(f"Модель {name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value}")
            except Exception as e:
                print(f"Ошибка при обучении модели {name}: {e}")
        
        if not results:
            print("Не удалось обучить ни одну модель")
            return None, None, None
            
        # Находим лучшую модель по метрике R2
        best_model_name = max(results, key=lambda x: results[x]['metrics']['r2'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]['metrics']
        
        print(f"Лучшая модель: {best_model_name}, R2: {best_metrics['r2']:.4f}")
        
        return best_model, best_metrics, best_model_name
    
    def train(self, df, optimize=True, feature_engineering=True, handle_outliers_method='winsorize'):
        """Обучение модели на исторических данных"""
        if df.empty or len(df) < 10:
            print("Недостаточно данных для обучения модели")
            return False
        
        print(f"Размер исходных данных: {df.shape}")
        
        # Сохраняем оригинальные данные для статистики
        self.data_properties = {
            'original_length': len(df),
            'min_date': df['date'].min().strftime('%Y-%m-%d'),
            'max_date': df['date'].max().strftime('%Y-%m-%d'),
            'min_value': df['value'].min(),
            'max_value': df['value'].max(),
            'mean_value': df['value'].mean(),
            'std_value': df['value'].std()
        }
        
        # Обработка выбросов
        if handle_outliers_method:
            df = self.handle_outliers(df, 'value', handle_outliers_method)
        
        # Дифференцирование, если необходимо
        df, self.is_differenced = self.difference_series(df)
        
        # Декомпозиция временного ряда для анализа
        trend, seasonal, residual = self.decompose_time_series(df)
        
        # Выбираем метод обучения в зависимости от данных
        if len(df) >= 30:  # Достаточно данных для ARIMA и ML
            print("Используем комбинированный подход: ARIMA + ML")
            
            # Разделяем данные на обучающую и тестовую выборки (80/20)
            train_size = int(len(df) * 0.8)
            df_train = df.iloc[:train_size].copy()
            df_test = df.iloc[train_size:].copy()
            
            # 1. Строим ARIMA модель
            arima_model = self.build_arima_model(df_train)
            
            # 2. Подготавливаем ML модель
            X_train, y_train, features = self.prepare_features(df_train, engineer_features=feature_engineering)
            
            if X_train is None:
                print("Не удалось подготовить признаки для ML модели")
                # Пробуем использовать только ARIMA
                if arima_model is not None:
                    self.model = arima_model
                    self.model_type = 'ARIMA'
                    self.best_model_type = 'ARIMA'
                    
                    # Оцениваем модель на тестовых данных
                    test_ts = df_test.set_index('date')['value']
                    arima_forecast = arima_model.get_forecast(steps=len(df_test))
                    y_pred = arima_forecast.predicted_mean.values
                    
                    # Рассчитываем метрики
                    self.metrics = self._calculate_metrics(test_ts.values, y_pred)
                    print("Используем только ARIMA модель")
                    print(f"Метрики ARIMA: {self.metrics}")
                    
                    self._save_model(features=[])
                    return True
                else:
                    print("Не удалось создать ни ARIMA, ни ML модель")
                    return False
            
            # Нормализуем признаки
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Подготавливаем тестовые данные для ML
            X_test, y_test, _ = self.prepare_features(df_test, engineer_features=feature_engineering)
            
            if X_test is None:
                print("Не удалось подготовить тестовые данные")
                return False
                
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучаем ML модели
            ml_model, ml_metrics, ml_model_name = self.train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            if ml_model is None:
                if arima_model is not None:
                    # Используем только ARIMA
                    self.model = arima_model
                    self.model_type = 'ARIMA'
                    self.best_model_type = 'ARIMA'
                    
                    # Оцениваем ARIMA на тесте
                    test_ts = df_test.set_index('date')['value']
                    arima_forecast = arima_model.get_forecast(steps=len(df_test))
                    y_pred = arima_forecast.predicted_mean.values
                    
                    self.metrics = self._calculate_metrics(test_ts.values, y_pred)
                    print(f"Метрики ARIMA: {self.metrics}")
                else:
                    print("Не удалось создать модели")
                    return False
            else:
                # У нас есть и ARIMA и ML модель
                if arima_model is not None:
                    # Оцениваем ARIMA на тесте
                    test_ts = df_test.set_index('date')['value']
                    arima_forecast = arima_model.get_forecast(steps=len(df_test))
                    arima_pred = arima_forecast.predicted_mean.values
                    
                    arima_metrics = self._calculate_metrics(test_ts.values, arima_pred)
                    print(f"Метрики ARIMA: {arima_metrics}")
                    
                    # Сравниваем метрики ARIMA и ML
                    if arima_metrics['r2'] > ml_metrics['r2']:
                        print("ARIMA модель показала лучшие результаты")
                        self.model = arima_model
                        self.model_type = 'ARIMA'
                        self.best_model_type = 'ARIMA'
                        self.metrics = arima_metrics
                    else:
                        print("ML модель показала лучшие результаты")
                        self.model = ml_model
                        self.model_type = 'ML'
                        self.best_model_type = ml_model_name
                        self.metrics = ml_metrics
                        self.feature_names = features
                else:
                    # Используем только ML модель
                    self.model = ml_model
                    self.model_type = 'ML'
                    self.best_model_type = ml_model_name
                    self.metrics = ml_metrics
                    self.feature_names = features
        else:
            # Недостаточно данных для ARIMA, используем только ML
            print("Недостаточно данных для ARIMA, используем только ML")
            
            # Разделяем данные на обучающую и тестовую выборки
            train_size = max(int(len(df) * 0.8), len(df) - 3)  # Оставляем минимум 3 точки для теста
            df_train = df.iloc[:train_size].copy()
            df_test = df.iloc[train_size:].copy() if train_size < len(df) else None
            
            # Подготавливаем признаки
            X_train, y_train, features = self.prepare_features(df_train, engineer_features=feature_engineering)
            
            if X_train is None:
                print("Не удалось подготовить признаки")
                return False
                
            # Нормализуем признаки
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Для очень малых выборок используем Ridge регрессию
            if len(df) < 20:
                print("Малая выборка, используем Ridge регрессию")
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                
                self.model = model
                self.model_type = 'ML'
                self.best_model_type = 'Ridge'
                self.feature_names = features
                
                if df_test is not None and not df_test.empty:
                    # Оцениваем на тестовых данных
                    X_test, y_test, _ = self.prepare_features(df_test, engineer_features=feature_engineering)
                    
                    if X_test is not None:
                        X_test_scaled = self.scaler.transform(X_test)
                        y_pred = model.predict(X_test_scaled)
                        self.metrics = self._calculate_metrics(y_test, y_pred)
                    else:
                        # Оцениваем на обучающих данных
                        y_pred = model.predict(X_train_scaled)
                        self.metrics = self._calculate_metrics(y_train, y_pred)
                        print("Внимание: метрики рассчитаны на обучающей выборке")
                else:
                    # Оцениваем на обучающих данных
                    y_pred = model.predict(X_train_scaled)
                    self.metrics = self._calculate_metrics(y_train, y_pred)
                    print("Внимание: метрики рассчитаны на обучающей выборке")
            else:
                # Достаточно данных для ML, но не для ARIMA
                if df_test is not None and not df_test.empty:
                    X_test, y_test, _ = self.prepare_features(df_test, engineer_features=feature_engineering)
                    
                    if X_test is not None:
                        X_test_scaled = self.scaler.transform(X_test)
                        
                        # Обучаем и сравниваем модели
                        ml_model, ml_metrics, ml_model_name = self.train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test)
                        
                        if ml_model is not None:
                            self.model = ml_model
                            self.model_type = 'ML'
                            self.best_model_type = ml_model_name
                            self.metrics = ml_metrics
                            self.feature_names = features
                        else:
                            # Используем простую Ridge регрессию
                            model = Ridge(alpha=1.0)
                            model.fit(X_train_scaled, y_train)
                            
                            y_pred = model.predict(X_test_scaled)
                            self.model = model
                            self.model_type = 'ML'
                            self.best_model_type = 'Ridge'
                            self.metrics = self._calculate_metrics(y_test, y_pred)
                            self.feature_names = features
                    else:
                        # Не удалось подготовить тестовые данные
                        model = Ridge(alpha=1.0)
                        model.fit(X_train_scaled, y_train)
                        
                        self.model = model
                        self.model_type = 'ML'
                        self.best_model_type = 'Ridge'
                        self.feature_names = features
                        
                        # Оцениваем на обучающих данных
                        y_pred = model.predict(X_train_scaled)
                        self.metrics = self._calculate_metrics(y_train, y_pred)
                        print("Внимание: метрики рассчитаны на обучающей выборке")
                else:
                    # Используем Ridge регрессию и оцениваем на обучающей выборке
                    model = Ridge(alpha=1.0)
                    model.fit(X_train_scaled, y_train)
                    
                    self.model = model
                    self.model_type = 'ML'
                    self.best_model_type = 'Ridge'
                    self.feature_names = features
                    
                    # Оцениваем на обучающих данных
                    y_pred = model.predict(X_train_scaled)
                    self.metrics = self._calculate_metrics(y_train, y_pred)
                    print("Внимание: метрики рассчитаны на обучающей выборке")
        
        # Сохраняем модель
        self._save_model(features)
        
        print(f"Модель обучена и сохранена: {os.path.join(self.model_dir, f'model_{self.version}.joblib')}")
        print(f"Тип модели: {self.model_type}, подтип: {self.best_model_type}")
        print(f"Метрики: {self.metrics}")
        
        return True
    
    def _calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества модели"""
        metrics = {}
        
        # R-квадрат
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mask = y_true != 0
        if any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = float('inf')
        
        # MRE
        metrics['mre'] = np.mean(np.abs(y_true - y_pred) / np.mean(y_true)) * 100
        
        return metrics
    
    def _save_model(self, features):
        """Сохранение модели и связанных данных"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Пути для сохранения
        model_path = os.path.join(self.model_dir, f"model_{self.version}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.version}.joblib")
        metrics_path = os.path.join(self.model_dir, f"metrics_{self.version}.joblib")
        metadata_path = os.path.join(self.model_dir, f"metadata_{self.version}.joblib")
        
        # Сохраняем модель
        joblib.dump(self.model, model_path)
        
        # Сохраняем скалер, если это ML модель
        if self.model_type == 'ML':
            joblib.dump(self.scaler, scaler_path)
        
        # Сохраняем метрики
        joblib.dump(self.metrics, metrics_path)
        
        # Сохраняем метаданные
        metadata = {
            'version': self.version,
            'model_type': self.model_type,
            'best_model_type': self.best_model_type,
            'features': features,
            'is_differenced': self.is_differenced,
            'original_mean': self.original_mean,
            'data_properties': self.data_properties,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(metadata, metadata_path)
    
    def predict_next_days(self, df, days=7):
        """Прогнозирование курса на следующие n дней"""
        if self.model is None:
            print("Модель не обучена")
            return pd.DataFrame()
            
        if df.empty or len(df) < 5:
            print("Недостаточно данных для прогноза")
            return pd.DataFrame()
        
        # Получаем последнюю дату из данных
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Разные методы прогнозирования в зависимости от типа модели
        if self.model_type == 'ARIMA':
            return self._predict_arima(df, future_dates)
        else:  # ML модель
            return self._predict_ml(df, future_dates)
    
    def _predict_arima(self, df, future_dates):
        """Прогноз с использованием ARIMA модели"""
        try:
            # Устанавливаем индекс для временного ряда
            ts = df.set_index('date')['value']
            
            # Делаем прогноз на нужное количество дней
            forecast = self.model.get_forecast(steps=len(future_dates))
            predictions = forecast.predicted_mean
            
            # Формируем результат
            result_df = pd.DataFrame({
                'date': future_dates,
                'predicted_value': predictions.values
            })
            
            # Если использовалось дифференцирование, восстанавливаем значения
            if self.is_differenced and self.original_mean is not None:
                # Получаем последнее значение
                last_value = df['value_original'].iloc[-1] if 'value_original' in df.columns else df['value'].iloc[-1]
                
                # Интегрируем дифференцированный ряд
                for i in range(len(result_df)):
                    if i == 0:
                        result_df.loc[i, 'predicted_value'] = last_value + result_df.loc[i, 'predicted_value']
                    else:
                        result_df.loc[i, 'predicted_value'] = result_df.loc[i-1, 'predicted_value'] + result_df.loc[i, 'predicted_value']
            
            return result_df
        except Exception as e:
            print(f"Ошибка при прогнозировании с ARIMA: {e}")
            return pd.DataFrame()
    
    def _predict_ml(self, df, future_dates):
        """Прогноз с использованием ML модели"""
        # Загружаем необходимые данные
        metadata_path = os.path.join(self.model_dir, f"metadata_{self.version}.joblib")
        
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            features = metadata.get('features', self.feature_names)
            is_differenced = metadata.get('is_differenced', self.is_differenced)
            original_mean = metadata.get('original_mean', self.original_mean)
        else:
            features = self.feature_names
            is_differenced = self.is_differenced
            original_mean = self.original_mean
        
        if not features:
            print("Не удалось получить список признаков")
            return pd.DataFrame()
        
        # Применяем дифференцирование, если оно использовалось при обучении
        processed_df = df.copy()
        if is_differenced:
            processed_df, _ = self.difference_series(processed_df)
        
        # Инициализируем DataFrame для результатов
        predictions = []
        current_data = processed_df.copy()
        
        for future_date in future_dates:
            # Создаем запись для следующего дня
            next_row = pd.DataFrame({'date': [future_date], 'value': [0]})  # Временное значение
            
            # Расширяем данные с новой записью
            extended_data = pd.concat([current_data, next_row], ignore_index=True)
            
            # Добавляем необходимые признаки
            extended_data['days_from_start'] = (extended_data['date'] - extended_data['date'].min()).dt.days
            extended_data['day_of_week'] = extended_data['date'].dt.dayofweek
            extended_data['day_sin'] = np.sin(2 * np.pi * extended_data['day_of_week'] / 7)
            extended_data['day_cos'] = np.cos(2 * np.pi * extended_data['day_of_week'] / 7)
            
            extended_data['month'] = extended_data['date'].dt.month
            extended_data['month_sin'] = np.sin(2 * np.pi * extended_data['month'] / 12)
            extended_data['month_cos'] = np.cos(2 * np.pi * extended_data['month'] / 12)
            
            extended_data['quarter'] = extended_data['date'].dt.quarter
            
            # Добавляем лаги
            for i in range(1, 6):
                if f'lag_{i}' in features:
                    extended_data[f'lag_{i}'] = extended_data['value'].shift(i)
            
            # Добавляем изменения
            if 'return_1d' in features:
                extended_data['return_1d'] = extended_data['value'].pct_change(1)
            if 'return_5d' in features:
                extended_data['return_5d'] = extended_data['value'].pct_change(5)
            
            # Добавляем скользящие средние
            if 'ma3' in features and len(extended_data) >= 3:
                extended_data['ma3'] = extended_data['value'].rolling(window=3).mean()
            if 'ma5' in features and len(extended_data) >= 5:
                extended_data['ma5'] = extended_data['value'].rolling(window=5).mean()
            if 'ewma' in features and len(extended_data) >= 5:
                extended_data['ewma'] = extended_data['value'].ewm(span=5).mean()
            
            # Добавляем волатильность
            if 'volatility' in features and len(extended_data) >= 5:
                extended_data['volatility'] = extended_data['value'].rolling(window=5).std()
            
            # Получаем значения признаков для прогнозируемого дня
            # Используем только те признаки, которые были при обучении
            available_features = [f for f in features if f in extended_data.columns]
            
            # Проверяем, что есть все необходимые признаки
            if set(available_features) != set(features):
                missing_features = set(features) - set(available_features)
                print(f"Предупреждение: отсутствуют признаки {missing_features}")
                
                # Добавляем отсутствующие признаки с нулевыми значениями
                for feature in missing_features:
                    extended_data[feature] = 0
                
                # Обновляем список доступных признаков
                available_features = features
            
            # Проверяем, что последняя строка не содержит NaN
            last_row = extended_data.iloc[-1]
            if last_row[available_features].isnull().any():
                print("Предупреждение: в данных есть NaN значения")
                
                # Заполняем NaN значения средними
                for feature in available_features:
                    if pd.isnull(last_row[feature]):
                        extended_data.loc[extended_data.index[-1], feature] = extended_data[feature].mean()
            
            # Получаем признаки для прогноза
            X_next = extended_data.iloc[-1][available_features].values.reshape(1, -1)
            
            # Нормализуем признаки
            X_next_scaled = self.scaler.transform(X_next)
            
            # Делаем прогноз
            prediction = self.model.predict(X_next_scaled)[0]
            
            # Если использовалось дифференцирование, восстанавливаем оригинальные значения
            if is_differenced and original_mean is not None:
                # Для первого прогноза используем последнее значение из исходных данных
                if len(predictions) == 0:
                    last_actual_value = df['value_original'].iloc[-1] if 'value_original' in df.columns else df['value'].iloc[-1]
                    prediction = last_actual_value + prediction
                else:
                    # Для последующих прогнозов используем предыдущее предсказание
                    prediction = predictions[-1]['predicted_value'] + prediction
            
            # Сохраняем прогноз
            predictions.append({
                'date': future_date,
                'predicted_value': prediction
            })
            
            # Обновляем текущие данные для следующего прогноза
            current_data = extended_data.copy()
            current_data.loc[current_data.index[-1], 'value'] = prediction
        
        return pd.DataFrame(predictions)
    
    def load_model(self, version=None):
        """Загрузка сохраненной модели"""
        if version is None:
            # Если версия не указана, ищем последнюю
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_")]
            if not model_files:
                print("Нет сохраненных моделей")
                return False
            
            model_files.sort(reverse=True)
            version = model_files[0].replace("model_", "").replace(".joblib", "")
        
        model_path = os.path.join(self.model_dir, f"model_{version}.joblib")
        metadata_path = os.path.join(self.model_dir, f"metadata_{version}.joblib")
        metrics_path = os.path.join(self.model_dir, f"metrics_{version}.joblib")
        
        if not os.path.exists(model_path):
            print(f"Модель версии {version} не найдена")
            return False
        
        # Загружаем модель
        self.model = joblib.load(model_path)
        self.version = version
        
        # Загружаем метаданные если есть
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.model_type = metadata.get('model_type', 'ML')
            self.best_model_type = metadata.get('best_model_type', 'Unknown')
            self.feature_names = metadata.get('features', [])
            self.is_differenced = metadata.get('is_differenced', False)
            self.original_mean = metadata.get('original_mean', None)
            self.data_properties = metadata.get('data_properties', {})
        else:
            self.model_type = 'ML'
            self.best_model_type = 'Unknown'
        
        # Загружаем скалер если это ML модель
        if self.model_type == 'ML':
            scaler_path = os.path.join(self.model_dir, f"scaler_{version}.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print("Предупреждение: файл скалера не найден")
        
        # Загружаем метрики
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {}
        
        print(f"Загружена модель версии {version}, тип: {self.model_type}, подтип: {self.best_model_type}")
        return True
    
    def get_model_version(self):
        """Получение текущей версии модели"""
        return self.version
    
    def get_model_metrics(self):
        """Получение метрик качества модели"""
        return self.metrics
    
    def get_model_info(self):
        """Получение информации о модели"""
        return {
            'version': self.version,
            'model_type': self.model_type,
            'best_model_type': self.best_model_type,
            'metrics': self.metrics,
            'features': self.feature_names,
            'is_differenced': self.is_differenced,
            'data_properties': self.data_properties
        }