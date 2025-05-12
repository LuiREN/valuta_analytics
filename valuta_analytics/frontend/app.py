import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Настройка страницы
st.set_page_config(
    page_title="ВалютАналитика",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Установка API URL
API_URL = os.environ.get("API_URL", "http://backend:8000")

# Базовые стили CSS (упрощенные для ускорения)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .header {
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    .header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 0.75rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        flex: 1;
    }
    .metric-title {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
    }
    .positive {
        color: #10b981;
    }
    .negative {
        color: #ef4444;
    }
    .neutral {
        color: #6b7280;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .metrics-table th {
        background-color: #f8f9fa;
        text-align: left;
        padding: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .metrics-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .model-quality {
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        display: inline-block;
        font-weight: 600;
    }
    .model-quality.good {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
    }
    .model-quality.medium {
        background-color: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
    }
    .model-quality.poor {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Функции для работы с API
@st.cache_data(ttl=300)
def get_currencies():
    """Получение списка всех доступных валют"""
    try:
        response = requests.get(f"{API_URL}/currencies")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Нет данных о валютах: {e}")
        return []

@st.cache_data(ttl=300)
def get_currency_history(currency_code, days=30):
    """Получение истории курса валюты"""
    try:
        response = requests.get(f"{API_URL}/currency/history/{currency_code}", params={"days": days})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Ошибка при получении истории валюты: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_statistics(currency_code, days=30):
    """Получение статистики по валюте"""
    try:
        response = requests.get(f"{API_URL}/currency/statistics/{currency_code}", params={"days": days})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении статистики: {e}")
        return None

def predict_currency(currency_code, days=7):
    """Прогнозирование курса валюты"""
    try:
        response = requests.post(f"{API_URL}/currency/predict/{currency_code}", params={"days": days})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении прогноза: {e}")
        return None

def update_current_data():
    """Обновление текущих курсов валют"""
    try:
        response = requests.post(f"{API_URL}/data/update/current")
        response.raise_for_status()
        # Принудительно обновляем кеш
        get_currencies.clear()
        get_last_update.clear()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обновлении данных: {e}")
        return None

def update_historical_data(start_date, end_date):
    """Обновление исторических данных"""
    try:
        response = requests.post(
            f"{API_URL}/data/update/historical",
            params={"start_date": start_date, "end_date": end_date}
        )
        response.raise_for_status()
        # Принудительно обновляем кеш
        get_currencies.clear()
        get_currency_history.clear()
        get_currency_statistics.clear()
        get_last_update.clear()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обновлении исторических данных: {e}")
        return {"status": "error", "message": str(e)}

def train_model(currency_code, days=60):
    """Обучение модели"""
    try:
        response = requests.post(f"{API_URL}/model/train/{currency_code}", params={"days": days})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обучении модели: {e}")
        return None

@st.cache_data(ttl=300)
def get_last_update():
    """Получение даты последнего обновления данных"""
    try:
        response = requests.get(f"{API_URL}/last-update")
        response.raise_for_status()
        return response.json().get("last_update", "Нет данных")
    except Exception as e:
        return f"Нет данных: {str(e)}"

@st.cache_data(ttl=300)
def get_weather(city, days_history=30, predict=True):
    """Получение данных о погоде для указанного города"""
    try:
        params = {"days_history": days_history, "predict": str(predict).lower()}
        response = requests.get(f"{API_URL}/weather/{city}", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении данных о погоде: {e}")
        return None

@st.cache_data(ttl=3600)
def get_popular_cities():
    """Получение списка популярных городов"""
    try:
        response = requests.get(f"{API_URL}/weather/popular-cities")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning("Не удалось загрузить список популярных городов")
        return []

# Функции для оценки моделей
def calculate_model_metrics(y_true, y_pred):
    """Расчет метрик качества модели"""
    metrics = {}
    
    # R-квадрат (коэффициент детерминации)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Средняя абсолютная ошибка (MAE)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Корень из среднеквадратичной ошибки (RMSE)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Средняя процентная ошибка (MAPE)
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Средняя относительная ошибка (MRE)
    metrics['mre'] = np.mean(np.abs(y_true - y_pred) / np.mean(y_true)) * 100
    
    return metrics

def evaluate_model_quality(r2_score):
    """Оценка качества модели на основе коэффициента детерминации"""
    if r2_score >= 0.7:
        return "good", "Хорошая"
    elif r2_score >= 0.5:
        return "medium", "Средняя"
    else:
        return "poor", "Низкая"

# Вспомогательные функции
def get_currency_emoji(code):
    """Получение эмодзи для валюты по коду"""
    emoji_dict = {
        'USD': '🇺🇸', 'EUR': '🇪🇺', 'GBP': '🇬🇧', 'JPY': '🇯🇵', 
        'CNY': '🇨🇳', 'CHF': '🇨🇭', 'AUD': '🇦🇺', 'CAD': '🇨🇦'
    }
    return emoji_dict.get(code, '💱')

def translate_day_of_week(day):
    """Перевод дня недели с английского на русский"""
    translations = {
        'Monday': 'Пн', 'Tuesday': 'Вт', 'Wednesday': 'Ср',
        'Thursday': 'Чт', 'Friday': 'Пт', 'Saturday': 'Сб', 'Sunday': 'Вс'
    }
    return translations.get(day, day)

def check_api_connection():
    """Проверка соединения с API"""
    try:
        response = requests.get(f"{API_URL}/")
        return True, response.status_code
    except Exception as e:
        return False, str(e)

# Основная функция приложения
def main():
    # Боковая панель
    with st.sidebar:
        st.title("ВалютАналитика")
        
        st.header("Управление данными")
        
        # Проверяем соединение с API
        api_status, status_details = check_api_connection()
        if api_status:
            st.success(f"API доступен (код: {status_details})")
        else:
            st.error(f"API недоступен: {status_details}")
            st.warning(f"URL API: {API_URL}")
        
        # Секция обновления данных
        with st.expander("Обновление данных"):
            if st.button("Загрузить текущие курсы", key="load_current"):
                result = update_current_data()
                if result and result.get("status") == "success":
                    st.success(result["message"])
                else:
                    st.warning(result.get("message", "Ошибка обновления"))
            
            st.subheader("Загрузка исторических данных")
            col1, col2 = st.columns(2)
            with col1:
                historical_start = st.date_input(
                    "Начальная дата",
                    datetime.now() - timedelta(days=30)
                )
            with col2:
                historical_end = st.date_input(
                    "Конечная дата",
                    datetime.now()
                )
            
            if st.button("Загрузить историю", key="load_history"):
                result = update_historical_data(
                    historical_start.strftime("%Y-%m-%d"),
                    historical_end.strftime("%Y-%m-%d")
                )
                if result and result.get("status") == "success":
                    st.success(result["message"])
                else:
                    st.warning(result.get("message", "Ошибка обновления"))
        
        # Секция модели
        with st.expander("Управление моделью"):
            currencies = get_currencies()
            currency_options = {curr["code"]: curr["name"] for curr in currencies} if currencies else {}
            
            if currency_options:
                selected_currency_code = st.selectbox(
                    "Валюта для обучения",
                    options=list(currency_options.keys()),
                    format_func=lambda x: f"{x} - {currency_options[x]}"
                )
                
                training_days = st.slider("Дни для обучения", 30, 365, 60)
                
                if st.button("Обучить модель", key="train_model"):
                    result = train_model(selected_currency_code, training_days)
                    if result and result.get("status") == "success":
                        st.success(result["message"])
                    else:
                        st.warning(result.get("message", "Ошибка обучения"))
            else:
                st.warning("Загрузите данные о валютах")
        
        # Информация о системе
        st.caption(f"Последнее обновление: {get_last_update()}")
        st.caption("Версия 2.0.0")
    
    # Основная часть приложения с вкладками
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Текущие курсы", "📈 Анализ валюты", "🔮 Прогнозирование", "🌤 Погода", "ℹ️ О проекте"])
    
    with tab1:
        st.header("Текущие курсы валют")
        st.caption(f"Последнее обновление: {get_last_update()}")
        
        # Получаем валюты
        currencies = get_currencies()
        
        if currencies:
            # Разделяем страницу на две колонки
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Основные валюты")
                
                # Выбираем основные валюты
                main_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'CHF']
                main_curr_data = [curr for curr in currencies if curr["code"] in main_currencies]
                
                for curr in main_curr_data:
                    st.metric(
                        label=f"{get_currency_emoji(curr['code'])} {curr['code']} - {curr['name']}",
                        value=f"{curr['rate']:.4f} ₽"
                    )
            
            with col2:
                st.subheader("Динамика основных валют")
                
                # Строим график
                fig = go.Figure()
                loaded_currencies = []
                
                for code in main_currencies:
                    currency_data = [curr for curr in currencies if curr["code"] == code]
                    if currency_data:
                        history = get_currency_history(code, 30)
                        if not history.empty:
                            fig.add_trace(go.Scatter(
                                x=history['date'],
                                y=history['value'],
                                mode='lines',
                                name=code
                            ))
                            loaded_currencies.append(code)
                
                if loaded_currencies:
                    fig.update_layout(
                        title="Курсы за последние 30 дней",
                        xaxis_title="Дата",
                        yaxis_title="Курс к рублю",
                        height=400,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Нет данных для отображения графика")
            
            # Таблица всех валют
            st.subheader("Все валюты")
            
            # Поиск по валютам
            search_term = st.text_input("Поиск валюты")
            
            # Создаем таблицу
            currency_df = pd.DataFrame([
                {"Код": curr["code"], "Название": curr["name"], "Курс к рублю": curr["rate"]}
                for curr in currencies if curr["rate"] is not None
            ])
            
            if search_term:
                filtered_df = currency_df[
                    currency_df["Код"].str.contains(search_term, case=False) | 
                    currency_df["Название"].str.contains(search_term, case=False)
                ]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(currency_df, use_container_width=True)
        else:
            st.warning("Нет данных о валютах. Загрузите данные через боковую панель.")
    
    with tab2:
        st.header("Анализ валюты")
        
        currencies = get_currencies()
        currency_options = {curr["code"]: curr["name"] for curr in currencies} if currencies else {}
        
        if currency_options:
            # Выбор валюты и параметров
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_code = st.selectbox(
                    "Выберите валюту",
                    options=list(currency_options.keys()),
                    format_func=lambda x: f"{x} - {currency_options[x]}"
                )
            
            with col2:
                days_to_analyze = st.slider("Количество дней", 7, 365, 30)
            
            # Получаем данные
            currency_history = get_currency_history(selected_code, days_to_analyze)
            stats = get_currency_statistics(selected_code, days_to_analyze)
            
            if not currency_history.empty and stats:
                # Отображаем статистику
                st.subheader("Статистика")
                
                # Метрики в колонках
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Текущий курс", 
                        f"{stats['current_rate']:.4f} ₽"
                    )
                
                with col2:
                    st.metric(
                        "Средний курс", 
                        f"{stats['avg_rate']:.4f} ₽"
                    )
                
                with col3:
                    st.metric(
                        "Минимум", 
                        f"{stats['min_rate']:.4f} ₽"
                    )
                
                with col4:
                    st.metric(
                        "Максимум", 
                        f"{stats['max_rate']:.4f} ₽"
                    )
                
                # Тренд и волатильность
                col1, col2 = st.columns(2)
                
                with col1:
                    trend_color = "positive" if stats['trend'] == "растущий" else "negative"
                    trend_icon = "↗" if stats['trend'] == "растущий" else "↘"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Тренд</div>
                        <div class="metric-value {trend_color}">
                            {trend_icon} {stats['trend'].capitalize()} ({stats['trend_value']:.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Волатильность</div>
                        <div class="metric-value neutral">
                            {stats['volatility']:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # График курса
                st.subheader("График курса")
                
                fig = go.Figure()
                
                # Основная линия
                fig.add_trace(go.Scatter(
                    x=currency_history['date'],
                    y=currency_history['value'],
                    mode='lines',
                    name=selected_code
                ))
                
                # Скользящее среднее (7 дней)
                currency_history['ma7'] = currency_history['value'].rolling(window=7).mean()
                if len(currency_history) >= 7:
                    fig.add_trace(go.Scatter(
                        x=currency_history['date'],
                        y=currency_history['ma7'],
                        mode='lines',
                        name='7-дневное среднее',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Динамика курса {selected_code} за {days_to_analyze} дней",
                    xaxis_title="Дата",
                    yaxis_title="Курс к рублю",
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Дополнительная визуализация
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Распределение изменений")
                    
                    # Рассчитываем дневные изменения
                    currency_history['pct_change'] = currency_history['value'].pct_change() * 100
                    
                    # Гистограмма изменений
                    fig = px.histogram(
                        currency_history.dropna(),
                        x='pct_change',
                        nbins=15,
                        labels={'pct_change': 'Дневное изменение, %'},
                        title="Распределение дневных изменений"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Статистика изменений")
                    
                    # Рассчитываем статистику
                    daily_changes = currency_history['pct_change'].dropna()
                    
                    # Дни роста/падения
                    up_days = sum(daily_changes > 0)
                    down_days = sum(daily_changes < 0)
                    flat_days = sum(daily_changes == 0)
                    
                    # Создаем круговую диаграмму
                    fig = px.pie(
                        values=[up_days, down_days, flat_days],
                        names=['Рост', 'Падение', 'Без изменений'],
                        title="Дни роста и падения"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Базовая статистика
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        avg_up = daily_changes[daily_changes > 0].mean() if any(daily_changes > 0) else 0
                        st.metric("Средний рост", f"+{avg_up:.2f}%")
                        
                        max_up = daily_changes.max()
                        st.metric("Макс. рост", f"+{max_up:.2f}%")
                    
                    with col2:
                        avg_down = daily_changes[daily_changes < 0].mean() if any(daily_changes < 0) else 0
                        st.metric("Среднее падение", f"{avg_down:.2f}%")
                        
                        max_down = daily_changes.min()
                        st.metric("Макс. падение", f"{max_down:.2f}%")
            else:
                st.warning(f"Недостаточно исторических данных для {selected_code}")
        else:
            st.warning("Нет данных о валютах. Загрузите данные через боковую панель.")
    
    with tab3:
        st.header("Прогнозирование курса валют")
        
        currencies = get_currencies()
        currency_options = {curr["code"]: curr["name"] for curr in currencies} if currencies else {}
        
        if currency_options:
            # Выбор валюты и параметров
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                selected_code = st.selectbox(
                    "Выберите валюту для прогноза",
                    options=list(currency_options.keys()),
                    format_func=lambda x: f"{x} - {currency_options[x]}",
                    key="predict_currency"
                )
            
            with col2:
                days_to_predict = st.slider("Дней прогноза", 1, 30, 7)
            
            with col3:
                predict_button = st.button("Сделать прогноз")
            
            if predict_button:
                with st.spinner('Создание прогноза...'):
                    prediction_result = predict_currency(selected_code, days_to_predict)
                    
                    if prediction_result:
                        history = get_currency_history(selected_code, 30)
                        
                        # Создаем DataFrame с прогнозами
                        predictions = pd.DataFrame([
                            {"date": pd.to_datetime(p["date"]), "predicted_value": p["predicted_value"]}
                            for p in prediction_result["predictions"]
                        ])
                        
                        # График прогноза
                        st.subheader("Прогноз курса")
                        
                        fig = go.Figure()
                        
                        # Добавляем историю
                        fig.add_trace(go.Scatter(
                            x=history['date'],
                            y=history['value'],
                            mode='lines',
                            name='Исторические данные'
                        ))
                        
                        # Добавляем прогноз
                        fig.add_trace(go.Scatter(
                            x=predictions['date'],
                            y=predictions['predicted_value'],
                            mode='lines+markers',
                            name='Прогноз',
                            line=dict(dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"Прогноз курса {selected_code} на {days_to_predict} дней",
                            xaxis_title="Дата",
                            yaxis_title="Курс к рублю",
                            height=400,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Таблица с прогнозом
                        st.subheader("Детали прогноза")
                        
                        # Преобразуем данные для отображения
                        df_display = pd.DataFrame([
                            {
                                "Дата": pd.to_datetime(p["date"]).strftime("%d.%m.%Y"),
                                "Прогноз курса": f"{p['predicted_value']:.4f} ₽",
                                "Изменение": f"{p['change_percent']:+.2f}%"
                            }
                            for p in prediction_result["predictions"]
                        ])
                        
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Оценка качества модели
                        st.subheader("Оценка качества модели")
                        
                        # Для оценки модели берем последние 7 дней исторических данных
                        # и сравниваем их с "прогнозом" модели на те же даты
                        eval_history = history.copy().iloc[-7:]
                        
                        if len(eval_history) >= 3:  # Минимум 3 точки для расчета метрик
                            # Получаем "прогнозы" для тех же дат, что и в eval_history
                            X_eval = np.arange(len(eval_history)).reshape(-1, 1)
                            
                            # Линейная регрессия для иллюстрации
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X_eval, eval_history['value'])
                            eval_predictions = model.predict(X_eval)
                            
                            # Рассчитываем метрики
                            metrics = calculate_model_metrics(eval_history['value'].values, eval_predictions)
                            
                            # Определяем качество модели
                            quality_class, quality_text = evaluate_model_quality(metrics['r2'])
                            
                            # Отображаем качество модели
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <span class="model-quality {quality_class}">
                                    Качество модели: {quality_text}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Таблица с метриками
                            st.markdown("""
                            <table class="metrics-table">
                                <tr>
                                    <th>Метрика</th>
                                    <th>Значение</th>
                                    <th>Описание</th>
                                </tr>
                                <tr>
                                    <td>R² (коэффициент детерминации)</td>
                                    <td>{:.4f}</td>
                                    <td>Доля дисперсии зависимой переменной, объясняемая моделью (0-1, чем ближе к 1, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MAE (средняя абсолютная ошибка)</td>
                                    <td>{:.4f}</td>
                                    <td>Среднее абсолютных значений ошибок (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>RMSE (корень из среднеквадратичной ошибки)</td>
                                    <td>{:.4f}</td>
                                    <td>Квадратный корень из среднего квадратов ошибок (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MAPE (средняя абсолютная процентная ошибка)</td>
                                    <td>{:.2f}%</td>
                                    <td>Средняя абсолютная ошибка в процентах (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MRE (средняя относительная ошибка)</td>
                                    <td>{:.2f}%</td>
                                    <td>Отношение средней ошибки к среднему значению (чем меньше, тем лучше)</td>
                                </tr>
                            </table>
                            """.format(
                                metrics['r2'],
                                metrics['mae'],
                                metrics['rmse'],
                                metrics['mape'],
                                metrics['mre']
                            ), unsafe_allow_html=True)
                            
                            # График сравнения факта и прогноза
                            st.subheader("Сравнение факта и модели")
                            
                            fig = go.Figure()
                            
                            # Фактические значения
                            fig.add_trace(go.Scatter(
                                x=eval_history['date'],
                                y=eval_history['value'],
                                mode='lines+markers',
                                name='Фактические значения'
                            ))
                            
                            # Прогнозные значения
                            fig.add_trace(go.Scatter(
                                x=eval_history['date'],
                                y=eval_predictions,
                                mode='lines+markers',
                                name='Прогнозные значения',
                                line=dict(dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f"Сравнение фактических и прогнозных значений",
                                xaxis_title="Дата",
                                yaxis_title="Курс к рублю",
                                height=400,
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Недостаточно данных для оценки качества модели")
                        
                        # Информация о модели
                        st.info(f"Модель: LinearRegression, версия: {prediction_result['model_version']}")
                    else:
                        st.error("Не удалось сделать прогноз")
        else:
            st.warning("Нет данных о валютах. Загрузите данные через боковую панель.")
    
    with tab4:
        st.header("Прогноз погоды")
        
        # Получаем список городов
        popular_cities = get_popular_cities()
        city_names = [city["name"] for city in popular_cities]
        
        # Выбор параметров
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            city = st.selectbox(
                "Выберите город",
                options=[""] + city_names,
                index=0
            )
            if not city:
                city = st.text_input("Или введите название города")
        
        with col2:
            days_history = st.slider(
                "Дней для анализа",
                min_value=7,
                max_value=60,
                value=30
            )
        
        with col3:
            predict = st.checkbox("ML-прогноз", value=True)
        
        with col4:
            get_weather_btn = st.button("Получить прогноз")
        
        if city and get_weather_btn:
            with st.spinner(f"Получение данных для {city}..."):
                weather_data = get_weather(city, days_history, predict)
            
            if weather_data:
                # Текущая погода
                st.subheader(f"Текущая погода в {weather_data['city']}")
                
                # Основные метрики
                current = weather_data['current']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Температура", 
                        f"{current['temperature']:.1f}°C", 
                        f"{current['feels_like'] - current['temperature']:+.1f}°C"
                    )
                
                with col2:
                    st.metric("Влажность", f"{current['humidity']}%")
                
                with col3:
                    st.metric("Давление", f"{current['pressure']} гПа")
                
                with col4:
                    st.metric("Ветер", f"{current['wind_speed']} м/с")
                
                st.info(f"**Описание**: {current['description']}")
                
                # Прогноз
                st.subheader("Прогноз погоды")
                
                forecast_tabs = st.tabs(["График", "Таблица", "Сравнение", "Метрики ML"])
                
                with forecast_tabs[0]:
                    # График прогноза температуры
                    df_forecast = pd.DataFrame([
                        {
                            "Дата": day['date'],
                            "Температура": day['temperature'],
                            "Источник": "OpenWeatherMap"
                        }
                        for day in weather_data['forecast']
                    ])
                    
                    # Добавляем ML прогноз
                    if 'predictions' in weather_data and weather_data['predictions']:
                        for pred in weather_data['predictions']:
                            df_forecast = pd.concat([df_forecast, pd.DataFrame([{
                                "Дата": pred['date'],
                                "Температура": pred['predicted_temperature'],
                                "Источник": "ML-модель"
                            }])], ignore_index=True)
                    
                    # Сортируем по дате
                    df_forecast['Дата'] = pd.to_datetime(df_forecast['Дата'])
                    df_forecast = df_forecast.sort_values('Дата')
                    
                    # Упрощенный формат даты
                    df_forecast['Дата_формат'] = df_forecast['Дата'].dt.strftime("%d.%m")
                    
                    # Строим график
                    fig = px.line(
                        df_forecast, 
                        x="Дата_формат", 
                        y="Температура", 
                        color="Источник",
                        markers=True,
                        title="Прогноз температуры"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Дата",
                        yaxis_title="Температура, °C",
                        height=350,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with forecast_tabs[1]:
                    # Таблица прогноза
                    forecast_data = []
                    for day in weather_data['forecast']:
                        # Преобразуем дату
                        date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
                        date_formatted = date_obj.strftime("%d.%m.%Y")
                        day_of_week = translate_day_of_week(date_obj.strftime("%A"))
                        
                        forecast_data.append({
                            "Дата": f"{date_formatted} ({day_of_week})",
                            "Температура": f"{day['temperature']:.1f}°C",
                            "Ощущается как": f"{day['feels_like']:.1f}°C",
                            "Влажность": f"{day['humidity']}%",
                            "Ветер": f"{day['wind_speed']} м/с",
                            "Описание": day['description'].capitalize()
                        })
                    
                    if forecast_data:
                        st.dataframe(pd.DataFrame(forecast_data), use_container_width=True)
                    else:
                        st.info("Прогноз на ближайшие дни недоступен")
                
                with forecast_tabs[2]:
                    # Сравнение прогнозов OpenWeatherMap и ML-модели
                    if 'predictions' in weather_data and weather_data['predictions']:
                        st.subheader("Сравнение прогнозов")
                        
                        # Создаем данные для сравнения
                        compare_data = {}
                        
                        # Добавляем прогнозы из API
                        for day in weather_data['forecast']:
                            date = day['date']
                            compare_data[date] = {"OpenWeatherMap": day['temperature']}
                        
                        # Добавляем ML прогнозы
                        for pred in weather_data['predictions']:
                            date = pred['date']
                            if date in compare_data:
                                compare_data[date]["ML-модель"] = pred['predicted_temperature']
                            else:
                                compare_data[date] = {"ML-модель": pred['predicted_temperature']}
                        
                        # Преобразуем в DataFrame
                        comparison_rows = []
                        for date, values in compare_data.items():
                            row = {"Дата": date}
                            row.update(values)
                            if "OpenWeatherMap" in values and "ML-модель" in values:
                                row["Разница"] = values["ML-модель"] - values["OpenWeatherMap"]
                            else:
                                row["Разница"] = None
                            comparison_rows.append(row)
                        
                        # Создаем DataFrame
                        comparison_df = pd.DataFrame(comparison_rows)
                        comparison_df["Дата"] = pd.to_datetime(comparison_df["Дата"])
                        comparison_df = comparison_df.sort_values("Дата")
                        
                        # Форматируем дату для отображения
                        formatted_df = comparison_df.copy()
                        formatted_df["Дата"] = formatted_df["Дата"].dt.strftime("%d.%m.%Y")
                        formatted_df["OpenWeatherMap"] = formatted_df["OpenWeatherMap"].apply(lambda x: f"{x:.1f}°C" if pd.notnull(x) else "Н/Д")
                        formatted_df["ML-модель"] = formatted_df["ML-модель"].apply(lambda x: f"{x:.1f}°C" if pd.notnull(x) else "Н/Д")
                        formatted_df["Разница"] = formatted_df["Разница"].apply(
                            lambda x: f"{x:+.1f}°C" if pd.notnull(x) else "Н/Д"
                        )
                        
                        # Отображаем таблицу сравнения
                        st.dataframe(
                            formatted_df[["Дата", "OpenWeatherMap", "ML-модель", "Разница"]],
                            use_container_width=True
                        )
                        
                        # Столбчатая диаграмма
                        st.subheader("Визуальное сравнение")
                        
                        fig = go.Figure()
                        
                        # Данные для графика
                        dates = comparison_df["Дата"].dt.strftime("%d.%m")
                        
                        # OpenWeatherMap
                        fig.add_trace(go.Bar(
                            x=dates,
                            y=comparison_df["OpenWeatherMap"],
                            name="OpenWeatherMap"
                        ))
                        
                        # ML-модель
                        fig.add_trace(go.Bar(
                            x=dates,
                            y=comparison_df["ML-модель"],
                            name="ML-модель"
                        ))
                        
                        fig.update_layout(
                            title="Сравнение прогнозов",
                            xaxis_title="Дата",
                            yaxis_title="Температура, °C",
                            barmode="group",
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("ML-прогноз недоступен. Включите опцию 'ML-прогноз'.")
                
                with forecast_tabs[3]:
                    # Оценка качества ML-модели для прогноза погоды
                    if 'predictions' in weather_data and weather_data['predictions'] and 'forecast' in weather_data:
                        st.subheader("Метрики качества ML-модели")
                        
                        # Сопоставляем даты прогнозов
                        owm_dates = {day['date']: day['temperature'] for day in weather_data['forecast']}
                        ml_dates = {pred['date']: pred['predicted_temperature'] for pred in weather_data['predictions']}
                        
                        # Находим общие даты для сравнения
                        common_dates = list(set(owm_dates.keys()) & set(ml_dates.keys()))
                        
                        if common_dates:
                            # Сортируем даты
                            common_dates.sort()
                            
                            # Получаем соответствующие значения прогнозов
                            owm_values = [owm_dates[date] for date in common_dates]
                            ml_values = [ml_dates[date] for date in common_dates]
                            
                            # Рассчитываем метрики для сравнения ML-модели с OpenWeatherMap
                            metrics = calculate_model_metrics(np.array(owm_values), np.array(ml_values))
                            
                            # Определяем качество модели
                            quality_class, quality_text = evaluate_model_quality(metrics['r2'])
                            
                            # Отображаем качество модели
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <span class="model-quality {quality_class}">
                                    Качество ML-модели: {quality_text}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Таблица с метриками
                            st.markdown("""
                            <table class="metrics-table">
                                <tr>
                                    <th>Метрика</th>
                                    <th>Значение</th>
                                    <th>Описание</th>
                                </tr>
                                <tr>
                                    <td>R² (коэффициент детерминации)</td>
                                    <td>{:.4f}</td>
                                    <td>Доля дисперсии зависимой переменной, объясняемая моделью (0-1, чем ближе к 1, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MAE (средняя абсолютная ошибка)</td>
                                    <td>{:.4f} °C</td>
                                    <td>Среднее абсолютных значений ошибок (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>RMSE (корень из среднеквадратичной ошибки)</td>
                                    <td>{:.4f} °C</td>
                                    <td>Квадратный корень из среднего квадратов ошибок (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MAPE (средняя абсолютная процентная ошибка)</td>
                                    <td>{:.2f}%</td>
                                    <td>Средняя абсолютная ошибка в процентах (чем меньше, тем лучше)</td>
                                </tr>
                                <tr>
                                    <td>MRE (средняя относительная ошибка)</td>
                                    <td>{:.2f}%</td>
                                    <td>Отношение средней ошибки к среднему значению (чем меньше, тем лучше)</td>
                                </tr>
                            </table>
                            """.format(
                                metrics['r2'],
                                metrics['mae'],
                                metrics['rmse'],
                                metrics['mape'],
                                metrics['mre']
                            ), unsafe_allow_html=True)
                            
                            # График отклонений
                            st.subheader("График отклонений")
                            
                            deviations = [ml - owm for ml, owm in zip(ml_values, owm_values)]
                            dates_fmt = [datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m") for date in common_dates]
                            
                            fig = go.Figure()
                            
                            # Отклонения
                            fig.add_trace(go.Bar(
                                x=dates_fmt,
                                y=deviations,
                                marker_color=['red' if d < 0 else 'green' for d in deviations]
                            ))
                            
                            # Нулевая линия
                            fig.add_shape(
                                type="line",
                                x0=0,
                                y0=0,
                                x1=1,
                                y1=0,
                                xref="paper",
                                line=dict(
                                    color="black",
                                    width=1,
                                    dash="dot",
                                )
                            )
                            
                            fig.update_layout(
                                title="Отклонение ML-модели от OpenWeatherMap",
                                xaxis_title="Дата",
                                yaxis_title="Отклонение, °C",
                                height=350
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Рекомендации по улучшению модели
                            st.subheader("Рекомендации по улучшению модели")
                            
                            if metrics['r2'] < 0.5:
                                st.warning("""
                                **Модель требует улучшения!**
                                
                                Рекомендации:
                                1. Увеличьте объем исторических данных для обучения
                                2. Добавьте дополнительные признаки (влажность, давление, сезонность)
                                3. Попробуйте более сложные алгоритмы (градиентный бустинг, нейронные сети)
                                4. Разделите данные по сезонам и обучите отдельные модели
                                """)
                            elif metrics['r2'] < 0.7:
                                st.info("""
                                **Модель требует доработки.**
                                
                                Рекомендации:
                                1. Увеличьте объем данных для обучения
                                2. Добавьте дополнительные признаки
                                3. Проведите тщательную настройку гиперпараметров
                                """)
                            else:
                                st.success("""
                                **Модель показывает хорошие результаты!**
                                
                                Для дальнейшего улучшения:
                                1. Периодически переобучайте модель на новых данных
                                2. Добавьте мониторинг дрейфа данных
                                """)
                        else:
                            st.warning("Недостаточно данных для сравнения прогнозов")
                    else:
                        st.warning("ML-прогноз недоступен. Включите опцию 'ML-прогноз'.")
                
                # Исторические данные
                if 'historical_data' in weather_data and weather_data['historical_data']:
                    st.subheader("Исторические данные")
                    
                    # Создаем DataFrame
                    hist_data = pd.DataFrame([{
                        "Дата": item['date'],
                        "Температура": item['temperature'],
                        "Влажность": item.get('humidity'),
                        "Давление": item.get('pressure'),
                        "Скорость ветра": item.get('wind_speed')
                    } for item in weather_data['historical_data']])
                    
                    # Преобразуем даты и сортируем
                    hist_data["Дата"] = pd.to_datetime(hist_data["Дата"])
                    hist_data = hist_data.sort_values("Дата")
                    
                    # График температуры
                    fig = px.line(
                        hist_data,
                        x="Дата",
                        y="Температура",
                        title=f"Историческая температура в {weather_data['city']}"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Дата",
                        yaxis_title="Температура, °C",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Базовая статистика
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Средняя температура", f"{hist_data['Температура'].mean():.1f}°C")
                    
                    with col2:
                        st.metric("Минимум", f"{hist_data['Температура'].min():.1f}°C")
                    
                    with col3:
                        st.metric("Максимум", f"{hist_data['Температура'].max():.1f}°C")
                    
                    with col4:
                        temp_range = hist_data['Температура'].max() - hist_data['Температура'].min()
                        st.metric("Размах", f"{temp_range:.1f}°C")
            else:
                st.error(f"Не удалось получить данные о погоде для '{city}'")
        else:
            st.info("Введите название города и нажмите 'Получить прогноз'")
    
    with tab5:
        st.header("О проекте")
        
        st.markdown("""
        ## ВалютАналитика
        
        Проект разработан для анализа и прогнозирования курсов валют на основе данных Центрального Банка РФ.
        
        ### Основные возможности:
        
        - **Сбор данных**: Автоматический сбор текущих и исторических курсов валют
        - **Анализ**: Статистический анализ динамики курсов, волатильности и трендов
        - **Прогнозирование**: Использование машинного обучения для прогнозирования курсов
        - **Визуализация**: Интерактивные графики и таблицы для представления информации
        - **Прогноз погоды**: Дополнительный модуль для анализа и прогноза погоды
        - **Метрики моделей**: Оценка качества моделей с помощью R², MAE, RMSE и др.
        
        ### Технологии:
        
        - **Бэкенд**: Python, FastAPI, SQLite, pandas, scikit-learn
        - **Фронтенд**: Streamlit, Plotly, HTML/CSS
        - **Инфраструктура**: Docker, Docker Compose
        
        ### Архитектура:
        
        - **Клиент-серверная архитектура**: Разделение на бэкенд (API) и фронтенд (пользовательский интерфейс)
        - **Бэкенд (FastAPI)**: API для работы с данными и моделями
        - **Фронтенд (Streamlit)**: Интерактивный веб-интерфейс
        
        ### Версия: 2.0.0 (2025)
        """)

# Запуск приложения
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ошибка при запуске приложения: {str(e)}")
        
        # Отображаем информацию о соединении
        st.info(f"URL API: {API_URL}")
        
        if st.button("Проверить соединение"):
            status, details = check_api_connection()
            if status:
                st.success(f"API доступен (код: {details})")
            else:
                st.error(f"API недоступен: {details}")
                st.info("Убедитесь, что сервер бэкенда запущен и доступен по указанному адресу.")