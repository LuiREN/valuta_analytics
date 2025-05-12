import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from datetime import datetime, timedelta
import joblib
import os
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

class ImprovedCurrencyPredictor:
    def __init__(self, model_dir='models'):
        """Инициализация улучшенного предиктора для прогнозирования курсов валют"""
        self.model_dir = model_dir
        # Создаем директорию для моделей, если она не существует
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model = None
        self.scaler = None
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}  # Словарь для хранения метрик модели
        self.best_model_type = None  # Тип лучшей модели
        self.feature_importances = None  # Важность признаков
        self.is_differenced = False  # Флаг, указывающий, были ли данные дифференцированы
        self.original_mean = None  # Среднее значение до дифференцирования
    
    def check_stationarity(self, df):
        """Проверка стационарности временного ряда с помощью теста Дики-Фуллера"""
        result = adfuller(df['value'].values)
        p_value = result[1]
        
        print(f"P-value: {p_value}")
        if p_value <= 0.05:
            print("Временной ряд стационарен (на уровне значимости 5%)")
            return True
        else:
            print("Временной ряд не стационарен, рекомендуется дифференцирование")
            return False
    
    def apply_differencing(self, df):
        """Применение дифференцирования для обеспечения стационарности"""
        if len(df) <= 1:
            return df, False
            
        # Сохраняем исходное среднее значение для последующего восстановления
        self.original_mean = df['value'].mean()
        
        # Создаем копию, чтобы не изменять исходный df
        diff_df = df.copy()
        
        # Рассчитываем разницы (первого порядка)
        diff_df['value_diff'] = diff_df['value'].diff()
        
        # Удаляем первую строку с NaN и восстанавливаем исходную дату
        diff_df = diff_df.dropna().reset_index(drop=True)
        
        # Заменяем исходные значения на дифференцированные
        diff_df['value_original'] = diff_df['value'].copy()
        diff_df['value'] = diff_df['value_diff']
        
        return diff_df, True
    
    def identify_outliers(self, df, column='value', method='zscore', threshold=3.0):
        """Выявление выбросов в данных"""
        if method == 'zscore':
            # Z-score метод
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = df[z_scores > threshold]
            return outliers, z_scores > threshold
        elif method == 'iqr':
            # IQR метод (межквартильный размах)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers, (df[column] < lower_bound) | (df[column] > upper_bound)
        else:
            raise ValueError("Неизвестный метод обнаружения выбросов")
    
    def handle_outliers(self, df, column='value', method='winsorize'):
        """Обработка выбросов в данных"""
        # Находим выбросы
        _, is_outlier = self.identify_outliers(df, column)
        
        if not any(is_outlier):
            return df  # Нет выбросов
            
        # Создаем копию для изменений
        clean_df = df.copy()
        
        if method == 'winsorize':
            # Ограничиваем экстремальные значения
            Q1 = df[column].quantile(0.05)  # 5-й процентиль
            Q3 = df[column].quantile(0.95)  # 95-й процентиль
            clean_df.loc[clean_df[column] < Q1, column] = Q1
            clean_df.loc[clean_df[column] > Q3, column] = Q3
        elif method == 'mean':
            # Заменяем выбросы средним значением
            mean_value = df[~is_outlier][column].mean()
            clean_df.loc[is_outlier, column] = mean_value
        elif method == 'median':
            # Заменяем выбросы медианой
            median_value = df[~is_outlier][column].median()
            clean_df.loc[is_outlier, column] = median_value
        elif method == 'remove':
            # Удаляем выбросы
            clean_df = df[~is_outlier].reset_index(drop=True)
            
        return clean_df
    
    def _prepare_features(self, df, engineer_features=True):
        """Улучшенная подготовка признаков для модели"""
        if df.empty:
            print("Нет данных для подготовки признаков")
            return None, None, None
        
        # Базовые признаки как в исходной модели
        # Создаем признак - день недели
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Создаем признаки для тренда
        df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Расширяем набор признаков, если разрешено инженерией признаков
        if engineer_features:
            # Циклические признаки для дня недели
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Признаки месяца
            df['month'] = df['date'].dt.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Сезонные признаки (зима, весна, лето, осень)
            df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
            df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
            df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
            df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
            
            # Признаки конца/начала месяца (часто влияют на валютные курсы)
            df['day_of_month'] = df['date'].dt.day
            df['end_of_month'] = (df['day_of_month'] >= 25).astype(int)
            df['start_of_month'] = (df['day_of_month'] <= 5).astype(int)
            
            # Квартал
            df['quarter'] = df['date'].dt.quarter
            
            # Дополнительные интегральные признаки на основе временных лагов
            
            # Скользящие средние разной глубины
            if len(df) >= 7:
                df['ma3'] = df['value'].rolling(window=3).mean()
                df['ma5'] = df['value'].rolling(window=5).mean()
                df['ma7'] = df['value'].rolling(window=7).mean()
            
            # Скользящие стандартные отклонения (волатильность)
            if len(df) >= 7:
                df['volatility3'] = df['value'].rolling(window=3).std()
                df['volatility7'] = df['value'].rolling(window=7).std()
            
            # Объем изменения (разница между максимумом и минимумом)
            if len(df) >= 5:
                df['range5'] = df['value'].rolling(window=5).max() - df['value'].rolling(window=5).min()
            
            # Моментум (изменение за определенный период)
            if len(df) >= 5:
                df['momentum3'] = df['value'] - df['value'].shift(3)
                df['momentum5'] = df['value'] - df['value'].shift(5)
            
            # Направление (положительное или отрицательное изменение)
            df['direction'] = (df['value'].diff() > 0).astype(int)
            
            # Величина изменения
            df['change'] = df['value'].diff()
            df['pct_change'] = df['value'].pct_change()
        
        # Сдвиги цен для учета предыдущих значений (лаги) - расширенная версия
        for i in range(1, 10):  # Увеличиваем количество лагов до 9
            df[f'lag_{i}'] = df['value'].shift(i)
        
        # Удаляем строки с NaN значениями (возникают из-за сдвигов)
        df = df.dropna()
        
        # Формируем списки признаков в зависимости от уровня инженерии
        if engineer_features:
            base_features = ['day_of_week', 'days_from_start']
            cyclic_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
            seasonal_features = ['is_winter', 'is_spring', 'is_summer', 'is_autumn']
            calendar_features = ['end_of_month', 'start_of_month', 'quarter']
            
            # Включаем только те признаки, которые были созданы выше
            ma_features = [col for col in ['ma3', 'ma5', 'ma7'] if col in df.columns]
            volatility_features = [col for col in ['volatility3', 'volatility7'] if col in df.columns]
            range_features = [col for col in ['range5'] if col in df.columns]
            momentum_features = [col for col in ['momentum3', 'momentum5'] if col in df.columns]
            direction_features = ['direction'] if 'direction' in df.columns else []
            change_features = [col for col in ['change', 'pct_change'] if col in df.columns]
            
            lag_features = [f'lag_{i}' for i in range(1, 10)]
            
            # Объединяем все группы признаков
            features = (
                base_features + cyclic_features + seasonal_features + calendar_features +
                ma_features + volatility_features + range_features + momentum_features +
                direction_features + change_features + lag_features
            )
        else:
            # Базовый набор признаков
            features = ['day_of_week', 'days_from_start'] + [f'lag_{i}' for i in range(1, 10)]
        
        # Отфильтровываем признаки, которых может не быть при нехватке данных
        features = [f for f in features if f in df.columns]
        
        # Проверяем на наличие признаков после фильтрации
        if not features:
            print("После фильтрации не осталось признаков")
            return None, None, None
        
        X = df[features].values
        y = df['value'].values
        
        return X, y, features
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """Обучение и сравнение различных моделей"""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        results = {}
        
        # Обучаем и оцениваем каждую модель
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Рассчитываем метрики
                metrics = self._calculate_metrics(y_test, y_pred)
                results[name] = {'model': model, 'metrics': metrics}
                
                print(f"Модель {name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value}")
                print()
            except Exception as e:
                print(f"Ошибка при обучении модели {name}: {e}")
        
        # Находим лучшую модель по R2
        best_model_name = max(results, key=lambda x: results[x]['metrics']['r2'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]['metrics']
        
        print(f"\nЛучшая модель: {best_model_name}")
        print(f"Метрики: {best_metrics}")
        
        return best_model, best_metrics, best_model_name
    
    def optimize_hyperparameters(self, X_train, y_train, model_type):
        """Оптимизация гиперпараметров для выбранной модели"""
        # Создаем разбиение для временных рядов
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Параметры для каждого типа модели
        param_grids = {
            'LinearRegression': {},  # Нет гиперпараметров для оптимизации
            'Ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 8]
            },
            'SVR': {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['linear', 'rbf']
            }
        }
        
        # Если модель не требует оптимизации или нет заданных параметров
        if model_type not in param_grids or not param_grids[model_type]:
            if model_type == 'LinearRegression':
                return LinearRegression()
            else:
                print(f"Нет параметров для оптимизации модели {model_type}")
                return None
        
        # Создаем базовую модель
        if model_type == 'Ridge':
            model = Ridge()
        elif model_type == 'Lasso':
            model = Lasso()
        elif model_type == 'ElasticNet':
            model = ElasticNet()
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
        elif model_type == 'GradientBoosting':
            model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'SVR':
            model = SVR()
        else:
            print(f"Неизвестный тип модели: {model_type}")
            return None
        
        # Выполняем поиск по сетке
        grid_search = GridSearchCV(
            model,
            param_grids[model_type],
            cv=tscv,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Лучшие параметры для {model_type}: {grid_search.best_params_}")
        print(f"Лучший результат: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train(self, df, optimize=True, feature_engineering=True, handle_outliers_method=None):
        """Улучшенное обучение модели на исторических данных"""
        if df.empty:
            print("Нет данных для обучения модели")
            return False
        
        print(f"Размер исходных данных: {df.shape}")
        
        # Проверка стационарности
        is_stationary = self.check_stationarity(df)
        
        # Применение дифференцирования, если ряд не стационарен
        if not is_stationary and len(df) > 10:  # Минимальный размер для дифференцирования
            print("Применение дифференцирования для обеспечения стационарности")
            df, is_differenced = self.apply_differencing(df)
            self.is_differenced = is_differenced
            
            if is_differenced:
                print(f"Размер данных после дифференцирования: {df.shape}")
        
        # Обработка выбросов, если указан метод
        if handle_outliers_method:
            print(f"Обработка выбросов методом {handle_outliers_method}")
            df = self.handle_outliers(df, 'value', handle_outliers_method)
            print(f"Размер данных после обработки выбросов: {df.shape}")
        
        # Разделяем данные на обучающую и тестовую выборки (80/20)
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size].copy()
        df_test = df.iloc[train_size:].copy()
        
        print(f"Размер обучающей выборки: {df_train.shape}")
        print(f"Размер тестовой выборки: {df_test.shape}")
        
        # Подготовка признаков для обучения с расширенным набором
        X_train, y_train, features = self._prepare_features(df_train, engineer_features=feature_engineering)
        
        if X_train is None:
            print("Не удалось подготовить признаки для обучения")
            return False
        
        print(f"Количество признаков: {len(features)}")
        print(f"Признаки: {features}")
        
        # Нормализация данных
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Оценка на тестовой выборке, если есть достаточно данных
        if len(df_test) > 5:
            X_test, y_test, _ = self._prepare_features(df_test, engineer_features=feature_engineering)
            
            if X_test is None:
                print("Не удалось подготовить признаки для тестирования")
                return False
                
            X_test_scaled = self.scaler.transform(X_test)
            
            # Сравнение различных моделей
            print("Сравнение различных моделей:")
            best_model, best_metrics, best_model_name = self.evaluate_models(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            if optimize and best_model_name != 'LinearRegression':
                print(f"Оптимизация гиперпараметров для модели {best_model_name}")
                optimized_model = self.optimize_hyperparameters(X_train_scaled, y_train, best_model_name)
                
                if optimized_model:
                    # Проверяем, лучше ли оптимизированная модель
                    y_pred = optimized_model.predict(X_test_scaled)
                    optimized_metrics = self._calculate_metrics(y_test, y_pred)
                    
                    print(f"Метрики оптимизированной модели: {optimized_metrics}")
                    print(f"Метрики базовой модели: {best_metrics}")
                    
                    if optimized_metrics['r2'] > best_metrics['r2']:
                        print("Оптимизированная модель лучше, используем её")
                        self.model = optimized_model
                        self.metrics = optimized_metrics
                    else:
                        print("Базовая модель лучше, используем её")
                        self.model = best_model
                        self.metrics = best_metrics
                else:
                    self.model = best_model
                    self.metrics = best_metrics
            else:
                self.model = best_model
                self.metrics = best_metrics
            
            self.best_model_type = best_model_name
            
            # Сохраняем данные о важности признаков, если модель поддерживает это
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = dict(zip(features, self.model.feature_importances_))
                # Выводим самые важные признаки
                print("\nВажность признаков:")
                for feature, importance in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True):
                    print(f"{feature}: {importance:.4f}")
            
            # Вывод метрик модели
            print("\nМетрики лучшей модели:")
            for metric_name, value in self.metrics.items():
                print(f"{metric_name}: {value}")
                
        else:
            # Если недостаточно данных для тестирования, используем кросс-валидацию
            print("Недостаточно данных для тестовой выборки, используем обучающую выборку")
            
            # Обучаем простую модель
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            # Оценка на обучающей выборке (для справки)
            y_pred = self.model.predict(X_train_scaled)
            self.metrics = self._calculate_metrics(y_train, y_pred)
            self.best_model_type = 'LinearRegression'
            
            print("\nМетрики модели (на обучающих данных):")
            for metric_name, value in self.metrics.items():
                print(f"{metric_name}: {value}")
        
        # Сохранение модели и связанных данных
        self._save_model(features)
        
        print(f"Модель обучена и сохранена: {os.path.join(self.model_dir, f'model_{self.version}.joblib')}")
        return True
    
    def _calculate_metrics(self, y_true, y_pred):
        """Расчет расширенного набора метрик качества модели"""
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
            metrics['mape'] = float('inf')
        
        # Средняя относительная ошибка (MRE)
        metrics['mre'] = np.mean(np.abs(y_true - y_pred) / np.mean(y_true)) * 100
        
        # Средняя ошибка (ME) - смещение
        metrics['me'] = np.mean(y_true - y_pred)
        
        # Максимальная ошибка
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        return metrics
    
    def _save_model(self, features):
        """Сохранение модели и всех связанных данных"""
        # Создаем директорию, если её нет
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Пути для сохранения
        model_path = os.path.join(self.model_dir, f"model_{self.version}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.version}.joblib")
        metrics_path = os.path.join(self.model_dir, f"metrics_{self.version}.joblib")
        features_path = os.path.join(self.model_dir, f"features_{self.version}.joblib")
        metadata_path = os.path.join(self.model_dir, f"metadata_{self.version}.joblib")
        
        # Сохраняем модель и стандартизатор
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Сохраняем метрики
        joblib.dump(self.metrics, metrics_path)
        
        # Сохраняем список признаков
        joblib.dump(features, features_path)
        
        # Сохраняем метаданные
        metadata = {
            'version': self.version,
            'model_type': self.best_model_type,
            'feature_importances': self.feature_importances,
            'is_differenced': self.is_differenced,
            'original_mean': self.original_mean,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(metadata, metadata_path)
    
    def predict_next_days(self, df, days=7):
        """Улучшенное прогнозирование курса на следующие n дней"""
        if self.model is None or df.empty:
            print("Модель не обучена или нет данных для прогноза")
            return pd.DataFrame()
        
        # Загружаем признаки если они были сохранены
        features_path = os.path.join(self.model_dir, f"features_{self.version}.joblib")
        metadata_path = os.path.join(self.model_dir, f"metadata_{self.version}.joblib")
        
        if os.path.exists(features_path):
            features = joblib.load(features_path)
        else:
            # Подготавливаем текущие данные для получения списка признаков
            _, _, features = self._prepare_features(df)
            if features is None:
                print("Ошибка при подготовке признаков для прогноза")
                return pd.DataFrame()
        
        # Загружаем метаданные если они были сохранены
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            is_differenced = metadata.get('is_differenced', False)
            original_mean = metadata.get('original_mean', None)
        else:
            is_differenced = self.is_differenced
            original_mean = self.original_mean
        
        # Применяем дифференцирование к данным, если оно использовалось при обучении
        if is_differenced:
            print("Применение дифференцирования для прогноза")
            df, _ = self.apply_differencing(df)
        
        # Будем прогнозировать на указанное количество дней вперед
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        predictions = []
        current_data = df.copy()
        
        for future_date in future_dates:
            # Создаем запись для следующего дня
            next_row = pd.DataFrame({'date': [future_date], 'value': [0]})  # Временное значение
            
            # Расширяем данные с новой записью
            extended_data = pd.concat([current_data, next_row], ignore_index=True)
            
            # Подготовка признаков для данных с добавленным следующим днем
            # Используем инженерию признаков (True)
            extended_data['day_of_week'] = extended_data['date'].dt.dayofweek
            extended_data['days_from_start'] = (extended_data['date'] - extended_data['date'].min()).dt.days
            
            # Добавляем месяц и другие циклические признаки
            extended_data['month'] = extended_data['date'].dt.month
            extended_data['month_sin'] = np.sin(2 * np.pi * extended_data['month'] / 12)
            extended_data['month_cos'] = np.cos(2 * np.pi * extended_data['month'] / 12)
            
            extended_data['day_sin'] = np.sin(2 * np.pi * extended_data['day_of_week'] / 7)
            extended_data['day_cos'] = np.cos(2 * np.pi * extended_data['day_of_week'] / 7)
            
            # Сезонные признаки
            extended_data['is_winter'] = ((extended_data['month'] == 12) | (extended_data['month'] <= 2)).astype(int)
            extended_data['is_spring'] = ((extended_data['month'] >= 3) & (extended_data['month'] <= 5)).astype(int)
            extended_data['is_summer'] = ((extended_data['month'] >= 6) & (extended_data['month'] <= 8)).astype(int)
            extended_data['is_autumn'] = ((extended_data['month'] >= 9) & (extended_data['month'] <= 11)).astype(int)
            
            # Признаки конца/начала месяца
            extended_data['day_of_month'] = extended_data['date'].dt.day
            extended_data['end_of_month'] = (extended_data['day_of_month'] >= 25).astype(int)
            extended_data['start_of_month'] = (extended_data['day_of_month'] <= 5).astype(int)
            
            # Квартал
            extended_data['quarter'] = extended_data['date'].dt.quarter
            
            # Добавляем лаги
            for i in range(1, 10):
                extended_data[f'lag_{i}'] = extended_data['value'].shift(i)
            
            # Добавляем скользящие средние, если достаточно данных
            if len(extended_data) >= 7:
                extended_data['ma3'] = extended_data['value'].rolling(window=3).mean()
                extended_data['ma5'] = extended_data['value'].rolling(window=5).mean()
                extended_data['ma7'] = extended_data['value'].rolling(window=7).mean()
                
                extended_data['volatility3'] = extended_data['value'].rolling(window=3).std()
                extended_data['volatility7'] = extended_data['value'].rolling(window=7).std()
                
                if len(extended_data) >= 5:
                    extended_data['range5'] = extended_data['value'].rolling(window=5).max() - extended_data['value'].rolling(window=5).min()
                    
                    extended_data['momentum3'] = extended_data['value'] - extended_data['value'].shift(3)
                    extended_data['momentum5'] = extended_data['value'] - extended_data['value'].shift(5)
            
            # Изменения
            extended_data['change'] = extended_data['value'].diff()
            extended_data['pct_change'] = extended_data['value'].pct_change()
            extended_data['direction'] = (extended_data['value'].diff() > 0).astype(int)
            
            # Получаем значения признаков для прогнозируемого дня
            # Отфильтровываем только те признаки, которые были использованы при обучении
            available_features = [f for f in features if f in extended_data.columns]
            
            # Проверяем, что у нас есть все необходимые признаки
            if len(available_features) < len(features):
                missing_features = set(features) - set(available_features)
                print(f"Предупреждение: отсутствуют некоторые признаки: {missing_features}")
            
            # Выбираем последнюю строку для прогноза
            X_next = extended_data.iloc[-1][available_features].values.reshape(1, -1)
            
            # Если не хватает признаков, дополняем нулями
            if X_next.shape[1] < len(features):
                X_next_full = np.zeros((1, len(features)))
                X_next_full[:, :X_next.shape[1]] = X_next
                X_next = X_next_full
            
            # Стандартизация признаков
            X_next_scaled = self.scaler.transform(X_next)
            
            # Делаем прогноз
            prediction = self.model.predict(X_next_scaled)[0]
            
            # Если использовалось дифференцирование, восстанавливаем оригинальное значение
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
        """Загрузка сохраненной модели по версии"""
        if version is None:
            # Если версия не указана, ищем последнюю
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_")]
            if not model_files:
                print("Нет сохраненных моделей")
                return False
            
            model_files.sort(reverse=True)
            version = model_files[0].replace("model_", "").replace(".joblib", "")
        
        model_path = os.path.join(self.model_dir, f"model_{version}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{version}.joblib")
        metrics_path = os.path.join(self.model_dir, f"metrics_{version}.joblib")
        features_path = os.path.join(self.model_dir, f"features_{version}.joblib")
        metadata_path = os.path.join(self.model_dir, f"metadata_{version}.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Файлы модели версии {version} не найдены")
            return False
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.version = version
        
        # Загружаем метрики, если они есть
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {}
        
        # Загружаем метаданные, если они есть
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_model_type = metadata.get('model_type')
            self.feature_importances = metadata.get('feature_importances')
            self.is_differenced = metadata.get('is_differenced', False)
            self.original_mean = metadata.get('original_mean')
        
        print(f"Загружена модель версии {version}")
        return True
    
    def get_model_version(self):
        """Получение текущей версии модели"""
        return self.version
    
    def get_model_metrics(self):
        """Получение метрик качества модели"""
        return self.metrics
    
    def get_model_info(self):
        """Получение подробной информации о модели"""
        info = {
            'version': self.version,
            'model_type': self.best_model_type,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances,
            'is_differenced': self.is_differenced
        }
        return info