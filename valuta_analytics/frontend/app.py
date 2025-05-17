import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
import json
import base64
from io import BytesIO
from scipy.stats import norm as stats_norm
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
        result = response.json()
        
        # Проверим, есть ли метрики модели в ответе
        if "model_metrics" not in result:
            result["model_metrics"] = {}  # Добавим пустой словарь, если метрик нет
        if "model_type" not in result:
            result["model_type"] = "Неизвестная модель"
            
        return result
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

def update_historical_data(start_date, end_date, currencies=None):
    """Обновление исторических данных"""
    try:
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Добавляем валюты в параметры, если они указаны
        if currencies:
            response = requests.post(
                f"{API_URL}/data/update/historical",
                params=params,
                json={"currencies": currencies}
            )
        else:
            response = requests.post(
                f"{API_URL}/data/update/historical",
                params=params
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
with st.expander("Обновление данных", expanded=True):  # Открываем по умолчанию для лучшей видимости
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Текущие курсы")
        
        # Более заметная кнопка со значком
        if st.button("🔄 Загрузить текущие курсы", key="load_current", use_container_width=True):
            with st.spinner('Загрузка текущих курсов...'):
                result = update_current_data()
                if result and result.get("status") == "success":
                    st.success(result["message"])
                else:
                    st.warning(result.get("message", "Ошибка обновления"))
    
    with col2:
        st.subheader("Последнее обновление")
        last_update = get_last_update()
        
        if last_update != "Нет данных":
            # Форматируем дату для удобного отображения
            try:
                last_update_date = datetime.strptime(last_update, "%Y-%m-%d")
                days_ago = (datetime.now() - last_update_date).days
                
                # Индикатор актуальности данных
                if days_ago == 0:
                    st.info(f"Обновлено сегодня ✅")
                elif days_ago == 1:
                    st.info(f"Обновлено вчера ⚠️")
                elif days_ago <= 7:
                    st.warning(f"Обновлено {days_ago} дней назад ⚠️")
                else:
                    st.error(f"Данные устарели! Последнее обновление: {days_ago} дней назад ❌")
            except:
                st.info(f"Последнее обновление: {last_update}")
        else:
            st.warning("Данные еще не были загружены")
    
    # Разделитель
    st.markdown("---")
    
    st.subheader("Загрузка исторических данных")
    
    # Выбор периода
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
    
    # Получаем список доступных валют для выбора
    currencies = get_currencies()
    
    if currencies:
        # Группируем валюты по категориям для более удобного выбора
        major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'CHF']
        currency_options = {}
        
        # Сначала основные валюты
        for curr in currencies:
            if curr["code"] in major_currencies:
                currency_options[curr["code"]] = f"{curr['code']} - {curr['name']}"
        
        # Затем остальные валюты
        for curr in currencies:
            if curr["code"] not in major_currencies:
                currency_options[curr["code"]] = f"{curr['code']} - {curr['name']}"
        
        # Выбор режима загрузки
        load_mode = st.radio(
            "Режим загрузки:",
            ["Все валюты", "Выбранные валюты", "Основные валюты"],
            horizontal=True
        )
        
        selected_currencies = []
        
        if load_mode == "Выбранные валюты":
            # Многоколоночный выбор валют
            num_cols = 3
            cols = st.columns(num_cols)
            
            # Опции поиска и выбора всех/снятия выбора
            search_term = st.text_input("🔍 Поиск валюты", placeholder="Введите код или название валюты")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Выбрать все", key="select_all"):
                    st.session_state.selected_currencies = list(currency_options.keys())
            with col2:
                if st.button("Снять выбор", key="deselect_all"):
                    st.session_state.selected_currencies = []
            
            # Инициализируем состояние выбранных валют, если его еще нет
            if 'selected_currencies' not in st.session_state:
                st.session_state.selected_currencies = major_currencies.copy()
            
            # Фильтруем валюты по поисковому запросу
            filtered_currencies = {}
            if search_term:
                for code, name in currency_options.items():
                    if search_term.lower() in name.lower():
                        filtered_currencies[code] = name
            else:
                filtered_currencies = currency_options
            
            # Отображаем валюты в несколько колонок с чекбоксами
            currency_items = list(filtered_currencies.items())
            items_per_col = (len(currency_items) + num_cols - 1) // num_cols
            
            for i, col in enumerate(cols):
                start_idx = i * items_per_col
                end_idx = min((i + 1) * items_per_col, len(currency_items))
                
                for code, name in currency_items[start_idx:end_idx]:
                    selected = code in st.session_state.selected_currencies
                    if col.checkbox(name, value=selected, key=f"curr_{code}"):
                        if code not in st.session_state.selected_currencies:
                            st.session_state.selected_currencies.append(code)
                    else:
                        if code in st.session_state.selected_currencies:
                            st.session_state.selected_currencies.remove(code)
            
            selected_currencies = st.session_state.selected_currencies
            
            # Показываем количество выбранных валют
            st.info(f"Выбрано валют: {len(selected_currencies)}")
            
        elif load_mode == "Основные валюты":
            selected_currencies = major_currencies
            st.info(f"Будут загружены основные валюты: {', '.join(selected_currencies)}")
    
    # Кнопка загрузки с улучшенным UX
    load_button = st.button("📥 Загрузить историю", key="load_history", use_container_width=True, type="primary")
    
    if load_button:
        with st.spinner('Загрузка исторических данных...'):
            # Параметры запроса в зависимости от выбора
            params = {
                "start_date": historical_start.strftime("%Y-%m-%d"),
                "end_date": historical_end.strftime("%Y-%m-%d")
            }
            
            # Добавляем выбранные валюты, если они указаны
            if load_mode != "Все валюты" and selected_currencies:
                result = update_historical_data(
                    historical_start.strftime("%Y-%m-%d"),
                    historical_end.strftime("%Y-%m-%d"),
                    selected_currencies
                )
            else:
                result = update_historical_data(
                    historical_start.strftime("%Y-%m-%d"),
                    historical_end.strftime("%Y-%m-%d")
                )
            
            if result and result.get("status") == "success":
                st.success(result["message"])
                # Принудительно обновляем кеш
                get_currencies.clear()
                get_currency_history.clear()
                get_currency_statistics.clear()
                get_last_update.clear()
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
    
    # ВКЛАДКА 1: Текущие курсы
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
    
    # ВКЛАДКА 2: Анализ валюты
    with tab2:
        st.header("Анализ валюты")
    
        currencies = get_currencies()
        currency_options = {curr["code"]: curr["name"] for curr in currencies} if currencies else {}
    
        if currency_options:
            # Улучшенный выбор валюты с поиском
            st.subheader("Выберите валюту для анализа")
        
            # Выбор валюты и параметров
            col1, col2, col3 = st.columns([2, 1, 1])
        
            with col1:
                # Поиск валюты
                search_query = st.text_input("🔍 Поиск валюты", placeholder="Введите код или название валюты")
            
                # Фильтруем валюты по поисковому запросу
                filtered_options = {}
                if search_query:
                    for code, name in currency_options.items():
                        if search_query.lower() in code.lower() or search_query.lower() in name.lower():
                            filtered_options[code] = name
                else:
                    filtered_options = currency_options
            
                # Создаем список опций для selectbox
                options_list = list(filtered_options.keys())
            
                if not options_list:
                    st.warning("По вашему запросу не найдено валют")
                    selected_code = None
                else:
                    selected_code = st.selectbox(
                        "Валюта",
                        options=options_list,
                        format_func=lambda x: f"{x} - {currency_options[x]}"
                    )
        
            with col2:
                days_to_analyze = st.slider("Период анализа", 7, 365, 30)
        
            with col3:
                chart_type = st.selectbox(
                    "Тип графика",
                    options=["Линия", "Свечи", "Бары"],
                    index=0
                )
        
            if selected_code:
                # Получаем данные
                currency_history = get_currency_history(selected_code, days_to_analyze)
                stats = get_currency_statistics(selected_code, days_to_analyze)
            
                if not currency_history.empty and stats:
                    # Показываем текущий курс большим текстом с изменением
                    if len(currency_history) > 1:
                        current_value = currency_history['value'].iloc[-1]
                        prev_value = currency_history['value'].iloc[-2]
                        change = ((current_value / prev_value) - 1) * 100
                        change_text = f"{change:+.2f}%"
                    else:
                        current_value = currency_history['value'].iloc[-1]
                        change_text = "N/A"
                
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 10px;">
                        <h2 style="margin-bottom: 5px;">{selected_code} - {currency_options[selected_code]}</h2>
                        <div style="display: flex; justify-content: center; align-items: center;">
                            <span style="font-size: 2.5rem; font-weight: bold; margin-right: 10px;">{current_value:.4f} ₽</span>
                            <span style="font-size: 1.2rem; color: {'green' if change_text.startswith('+') else 'red'};">{change_text}</span>
                        </div>
                        <p style="color: #6c757d; margin-top: 5px;">Курс на {currency_history['date'].iloc[-1].strftime('%d.%m.%Y')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                    # Отображаем статистику в виде карточек
                    st.subheader("Ключевые показатели")
                
                    # Метрики в колонках
                    col1, col2, col3, col4 = st.columns(4)
                
                    with col1:
                        st.metric(
                            "Средний курс", 
                            f"{stats['avg_rate']:.4f} ₽"
                        )
                
                    with col2:
                        min_date = datetime.strptime(stats['min_date'], "%Y-%m-%d").strftime("%d.%m.%Y")
                        st.metric(
                            "Минимум", 
                            f"{stats['min_rate']:.4f} ₽",
                            f"({min_date})"
                        )
                
                    with col3:
                        max_date = datetime.strptime(stats['max_date'], "%Y-%m-%d").strftime("%d.%m.%Y")
                        st.metric(
                            "Максимум", 
                            f"{stats['max_rate']:.4f} ₽",
                            f"({max_date})"
                        )
                
                    with col4:
                        st.metric(
                            "Волатильность", 
                            f"{stats['volatility']:.2f}%"
                        )
                
                    # Интерактивные графики
                    st.subheader("Динамика курса")
                
                    # Вкладки для различных визуализаций
                    chart_tabs = st.tabs(["Общий график", "Технический анализ", "Сезонный анализ", "Распределение"])
                
                    with chart_tabs[0]:
                        # Основной график курса (линейный, свечной или бары)
                        if chart_type == "Линия":
                            fig = go.Figure()
                        
                            # Основная линия
                            fig.add_trace(go.Scatter(
                                x=currency_history['date'],
                                y=currency_history['value'],
                                mode='lines',
                                name=selected_code,
                                line=dict(width=2, color='blue')
                            ))
                        
                            # Скользящие средние (если достаточно данных)
                            if len(currency_history) >= 7:
                                currency_history['ma7'] = currency_history['value'].rolling(window=7).mean()
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=currency_history['ma7'],
                                    mode='lines',
                                    name='7-дневное среднее',
                                    line=dict(dash='dash', color='orange')
                                ))
                        
                            if len(currency_history) >= 21:
                                currency_history['ma21'] = currency_history['value'].rolling(window=21).mean()
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=currency_history['ma21'],
                                    mode='lines',
                                    name='21-дневное среднее',
                                    line=dict(dash='dash', color='green')
                                ))
                    
                        elif chart_type == "Свечи":
                            # Подготовка данных для свечного графика
                            # Требуется дополнительная обработка для получения OHLC
                            # Для демонстрации используем дневные данные
                            # В реальности нужны внутридневные данные для точного OHLC
                        
                            # Сначала ресемплируем данные до дневных
                            currency_history['date'] = pd.to_datetime(currency_history['date'])
                            currency_history.set_index('date', inplace=True)
                        
                            # Создаем псевдо-OHLC для демонстрации
                            # В настоящем приложении здесь должны использоваться настоящие данные OHLC
                            pseudo_ohlc = pd.DataFrame()
                            pseudo_ohlc['open'] = currency_history['value']
                        
                            # Создаем псевдо high/low на основе volatility
                            volatility_factor = stats['volatility'] / 100 / 2
                            pseudo_ohlc['high'] = pseudo_ohlc['open'] * (1 + np.random.uniform(0, volatility_factor, len(pseudo_ohlc)))
                            pseudo_ohlc['low'] = pseudo_ohlc['open'] * (1 - np.random.uniform(0, volatility_factor, len(pseudo_ohlc)))
                        
                            # Close - это значение следующего дня
                            pseudo_ohlc['close'] = pseudo_ohlc['open'].shift(-1)
                            pseudo_ohlc.dropna(inplace=True)
                        
                            # Для демонстрации меняем close для создания разных свечей
                            for i in range(len(pseudo_ohlc)):
                                if np.random.random() > 0.5:
                                    pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('close')] = (
                                        pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('open')] + 
                                        np.random.uniform(-volatility_factor, volatility_factor) * 
                                        pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('open')]
                                    )
                        
                            # Исправляем high/low на основе open/close
                            for i in range(len(pseudo_ohlc)):
                                open_val = pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('open')]
                                close_val = pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('close')]
                                high_val = pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('high')]
                                low_val = pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('low')]
                            
                                high_val = max(high_val, open_val, close_val)
                                low_val = min(low_val, open_val, close_val)
                            
                                pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('high')] = high_val
                                pseudo_ohlc.iloc[i, pseudo_ohlc.columns.get_loc('low')] = low_val
                        
                            # Создаем свечной график
                            fig = go.Figure(data=[go.Candlestick(
                                x=pseudo_ohlc.index,
                                open=pseudo_ohlc['open'],
                                high=pseudo_ohlc['high'],
                                low=pseudo_ohlc['low'],
                                close=pseudo_ohlc['close'],
                                increasing_line_color='green',
                                decreasing_line_color='red'
                            )])
                        
                            # Добавляем скользящую среднюю
                            if len(pseudo_ohlc) >= 7:
                                ma7 = pseudo_ohlc['close'].rolling(window=7).mean()
                                fig.add_trace(go.Scatter(
                                    x=ma7.index,
                                    y=ma7,
                                    mode='lines',
                                    name='7-дневное среднее',
                                    line=dict(dash='dash', color='blue')
                                ))
                    
                        elif chart_type == "Бары":
                            # Создаем данные для барчарта
                            # Рассчитываем изменения
                            currency_history['change'] = currency_history['value'].diff()
                            currency_history['color'] = ['green' if x > 0 else 'red' for x in currency_history['change']]
                        
                            fig = go.Figure(data=[go.Bar(
                                x=currency_history['date'],
                                y=currency_history['value'],
                                marker_color=currency_history['color']
                            )])
                    
                        # Общие настройки графика
                        fig.update_layout(
                            title=f"Динамика курса {selected_code} за {days_to_analyze} дней",
                            xaxis_title="Дата",
                            yaxis_title="Курс к рублю",
                            height=500,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                    
                        # Добавляем инструменты технического анализа через кнопки
                        col1, col2, col3 = st.columns(3)
                    
                        with col1:
                            show_trend = st.checkbox("Показать тренд", value=True)
                            if show_trend:
                                # Добавляем линию тренда
                                x = np.array(range(len(currency_history))).reshape(-1, 1)
                                y = currency_history['value'].values
                            
                                model = LinearRegression()
                                model.fit(x, y)
                                trend = model.predict(x)
                            
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=trend,
                                    mode='lines',
                                    name='Тренд',
                                    line=dict(color='purple', width=2)
                                ))
                    
                        with col2:
                            show_bands = st.checkbox("Полосы Боллинджера", value=False)
                            if show_bands and len(currency_history) >= 10:
                                # Добавляем полосы Боллинджера
                                window = 20
                                if len(currency_history) < window:
                                    window = len(currency_history) // 2
                            
                                currency_history['ma'] = currency_history['value'].rolling(window=window).mean()
                                currency_history['std'] = currency_history['value'].rolling(window=window).std()
                            
                                upper_band = currency_history['ma'] + (currency_history['std'] * 2)
                                lower_band = currency_history['ma'] - (currency_history['std'] * 2)
                            
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=upper_band,
                                    mode='lines',
                                    name='Верхняя полоса',
                                    line=dict(color='rgba(0, 128, 0, 0.3)')
                                ))
                            
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=lower_band,
                                    mode='lines',
                                    name='Нижняя полоса',
                                    line=dict(color='rgba(255, 0, 0, 0.3)')
                                ))
                            
                                # Заполнение между полосами
                                fig.add_trace(go.Scatter(
                                    x=pd.concat([currency_history['date'], currency_history['date'][::-1]]),
                                    y=pd.concat([upper_band, lower_band[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(100, 100, 250, 0.1)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                    
                        with col3:
                            log_scale = st.checkbox("Логарифмическая шкала", value=False)
                            if log_scale:
                                fig.update_layout(yaxis_type="log")
                    
                        st.plotly_chart(fig, use_container_width=True)
                
                    with chart_tabs[1]:
                        # Технический анализ
                        st.subheader("Технический анализ")
                    
                        # Рассчитываем технические индикаторы
                        # RSI
                        if len(currency_history) >= 14:
                            delta = currency_history['value'].diff()
                            gain = delta.clip(lower=0)
                            loss = -delta.clip(upper=0)
                        
                            avg_gain = gain.rolling(window=14).mean()
                            avg_loss = loss.rolling(window=14).mean()
                        
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                        
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                # График RSI
                                fig = go.Figure()
                            
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=rsi,
                                    mode='lines',
                                    name='RSI',
                                    line=dict(color='purple')
                                ))
                            
                                # Добавляем линии уровней
                                fig.add_shape(
                                    type="line",
                                    x0=currency_history['date'].min(),
                                    y0=70,
                                    x1=currency_history['date'].max(),
                                    y1=70,
                                    line=dict(
                                        color="red",
                                        width=1,
                                        dash="dash",
                                    )
                                )
                            
                                fig.add_shape(
                                    type="line",
                                    x0=currency_history['date'].min(),
                                    y0=30,
                                    x1=currency_history['date'].max(),
                                    y1=30,
                                    line=dict(
                                        color="green",
                                        width=1,
                                        dash="dash",
                                    )
                                )
                            
                                fig.update_layout(
                                    title="Индекс относительной силы (RSI)",
                                    xaxis_title="Дата",
                                    yaxis_title="RSI",
                                    height=300,
                                    yaxis=dict(
                                        range=[0, 100]
                                    )
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                            
                                # Добавим интерпретацию RSI
                                current_rsi = rsi.dropna().iloc[-1]
                                if current_rsi > 70:
                                    st.warning(f"RSI = {current_rsi:.1f} (> 70) - возможно перекупленность")
                                elif current_rsi < 30:
                                    st.warning(f"RSI = {current_rsi:.1f} (< 30) - возможно перепроданность")
                                else:
                                    st.info(f"RSI = {current_rsi:.1f} (нейтральная зона)")
                            
                            with col2:
                                # MACD
                                ema12 = currency_history['value'].ewm(span=12, adjust=False).mean()
                                ema26 = currency_history['value'].ewm(span=26, adjust=False).mean()
                                macd = ema12 - ema26
                                signal = macd.ewm(span=9, adjust=False).mean()
                                histogram = macd - signal
                            
                                fig = go.Figure()
                            
                                # MACD линия
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=macd,
                                    mode='lines',
                                    name='MACD',
                                    line=dict(color='blue')
                                ))
                            
                                # Сигнальная линия
                                fig.add_trace(go.Scatter(
                                    x=currency_history['date'],
                                    y=signal,
                                    mode='lines',
                                    name='Сигнал',
                                    line=dict(color='red')
                                ))
                            
                                # Гистограмма
                                fig.add_trace(go.Bar(
                                    x=currency_history['date'],
                                    y=histogram,
                                    name='Гистограмма',
                                    marker_color=['green' if x > 0 else 'red' for x in histogram]
                                ))
                            
                                fig.update_layout(
                                    title="MACD (схождение/расхождение скользящих средних)",
                                    xaxis_title="Дата",
                                    yaxis_title="MACD",
                                    height=300
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                            
                                # Добавляем интерпретацию MACD
                                last_macd = macd.iloc[-1]
                                last_signal = signal.iloc[-1]
                            
                                if last_macd > last_signal:
                                    st.info("MACD > Сигнальная линия: возможен бычий тренд")
                                else:
                                    st.info("MACD < Сигнальная линия: возможен медвежий тренд")
                        else:
                            st.warning("Недостаточно данных для расчета технических индикаторов (необходимо минимум 14 дней)")
                
                    with chart_tabs[2]:
                        # Сезонный анализ
                        st.subheader("Сезонный анализ")
                    
                        # Проверяем, достаточно ли данных
                        if len(currency_history) >= 30:
                            # Добавляем день недели и месяц
                            currency_history['day_of_week'] = currency_history['date'].dt.dayofweek
                            currency_history['month'] = currency_history['date'].dt.month
                        
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                # Средний курс по дням недели
                                day_avg = currency_history.groupby('day_of_week')['value'].mean().reset_index()
                                day_avg['day_name'] = day_avg['day_of_week'].map({
                                    0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'
                                })
                            
                                fig = px.bar(
                                    day_avg,
                                    x='day_name',
                                    y='value',
                                    title="Средний курс по дням недели",
                                    color='value',
                                    color_continuous_scale='Viridis'
                                )
                            
                                fig.update_layout(
                                    xaxis_title="День недели",
                                    yaxis_title="Средний курс",
                                    height=300
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                        
                            with col2:
                                # Средний курс по месяцам
                                month_avg = currency_history.groupby('month')['value'].mean().reset_index()
                                month_avg['month_name'] = month_avg['month'].map({
                                    1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн',
                                    7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
                                })
                            
                                fig = px.bar(
                                    month_avg,
                                    x='month_name',
                                    y='value',
                                    title="Средний курс по месяцам",
                                    color='value',
                                    color_continuous_scale='Viridis'
                                )
                            
                                fig.update_layout(
                                    xaxis_title="Месяц",
                                    yaxis_title="Средний курс",
                                    height=300
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Недостаточно данных для сезонного анализа (необходимо минимум 30 дней)")
                
                    with chart_tabs[3]:
                        # Распределение изменений
                        st.subheader("Анализ изменений")
                    
                        # Рассчитываем дневные изменения
                        currency_history['pct_change'] = currency_history['value'].pct_change() * 100
                    
                        # Гистограмма изменений
                        fig = px.histogram(
                            currency_history.dropna(),
                            x='pct_change',
                            nbins=20,
                            title="Распределение дневных изменений",
                            color_discrete_sequence=['blue']
                        )
                    
                        # Добавляем кривую нормального распределения
                        mean = currency_history['pct_change'].mean()
                        std = currency_history['pct_change'].std()
                    
                        x = np.linspace(mean - 3*std, mean + 3*std, 100)
                        y = stats_norm.pdf(x, mean, std) * len(currency_history) * (currency_history['pct_change'].max() - currency_history['pct_change'].min()) / 20
                    
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name='Нормальное распределение',
                            line=dict(color='red')
                        ))
                    
                        fig.update_layout(
                        xaxis_title="Изменение, %",
                        yaxis_title="Частота",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Статистика изменений
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Статистика дней роста/падения
                        daily_changes = currency_history['pct_change'].dropna()
                        
                        up_days = sum(daily_changes > 0)
                        down_days = sum(daily_changes < 0)
                        flat_days = sum(daily_changes == 0)
                        
                        # Создаем круговую диаграмму дней роста/падения
                        fig = px.pie(
                            values=[up_days, down_days, flat_days],
                            names=['Рост', 'Падение', 'Без изменений'],
                            title="Дни роста и падения",
                            color_discrete_sequence=['green', 'red', 'gray']
                        )
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Таблица со статистикой
                        st.subheader("Статистика изменений")
                        
                        avg_up = daily_changes[daily_changes > 0].mean() if any(daily_changes > 0) else 0
                        avg_down = daily_changes[daily_changes < 0].mean() if any(daily_changes < 0) else 0
                        max_up = daily_changes.max()
                        max_down = daily_changes.min()
                        
                        stats_df = pd.DataFrame([
                            {"Показатель": "Средний рост", "Значение": f"+{avg_up:.2f}%"},
                            {"Показатель": "Среднее падение", "Значение": f"{avg_down:.2f}%"},
                            {"Показатель": "Максимальный рост", "Значение": f"+{max_up:.2f}%"},
                            {"Показатель": "Максимальное падение", "Значение": f"{max_down:.2f}%"},
                            {"Показатель": "Стандартное отклонение", "Значение": f"{std:.2f}%"},
                            {"Показатель": "Дней роста", "Значение": f"{up_days} ({up_days/len(daily_changes)*100:.1f}%)"},
                            {"Показатель": "Дней падения", "Значение": f"{down_days} ({down_days/len(daily_changes)*100:.1f}%)"}
                        ])
                        
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Добавляем экспорт данных
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Экспорт данных")
                    
                    # Опции формата экспорта
                    export_format = st.selectbox(
                        "Формат файла",
                        options=["CSV", "Excel", "JSON"],
                        index=0
                    )
                    
                    # Кнопка для экспорта
                    if st.button("📥 Экспортировать данные"):
                        # Подготовка данных для экспорта
                        export_df = currency_history.copy()
                        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
                        
                        if export_format == "CSV":
                            csv = export_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_code}_history.csv">Скачать CSV-файл</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        elif export_format == "Excel":
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                export_df.to_excel(writer, sheet_name='Data', index=False)
                            b64 = base64.b64encode(output.getvalue()).decode()
                            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{selected_code}_history.xlsx">Скачать Excel-файл</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        elif export_format == "JSON":
                            json_data = export_df.to_json(orient='records')
                            b64 = base64.b64encode(json_data.encode()).decode()
                            href = f'<a href="data:file/json;base64,{b64}" download="{selected_code}_history.json">Скачать JSON-файл</a>'
                            st.markdown(href, unsafe_allow_html=True)
                
                with col2:
                    # Быстрый переход к прогнозу
                    st.subheader("Прогноз курса")
                    st.info(f"Хотите узнать, как изменится курс {selected_code} в ближайшее время?")
                    
                    predict_days = st.slider("Количество дней для прогноза", 1, 30, 7, key="quick_predict_days")
                    
                    if st.button("🔮 Сделать быстрый прогноз", key="quick_predict"):
                        # Перенаправляем на вкладку прогнозирования
                        st.session_state.quick_predict_currency = selected_code
                        st.session_state.quick_predict_days = predict_days
                        st.session_state.active_tab = "Прогнозирование"
                        st.experimental_rerun()
            else:
                st.warning(f"Недостаточно исторических данных для {selected_code}")
        else:
            st.warning("Нет данных о валютах. Загрузите данные через боковую панель.")
    
    # ВКЛАДКА 3: Прогнозирование курса валют
    with tab3:
        st.header("Прогнозирование курса валют")
    
        currencies = get_currencies()
        currency_options = {curr["code"]: curr["name"] for curr in currencies} if currencies else {}
    
        if currency_options:
            # Основная секция прогнозирования
            st.subheader("Создать новый прогноз")
        
            # Добавляем боковую карточку с информацией о прогнозах
            info_col, predict_col = st.columns([1, 3])
        
            with info_col:
                st.markdown("""
                ### О прогнозах
            
                Наша система использует **передовые методы машинного обучения** и анализа временных рядов для прогнозирования курсов валют.
            
                **Как это работает:**
                1. Система анализирует исторические данные
                2. Выбирает лучшую модель для прогноза
                3. Формирует прогноз на указанный период
            
                **Надежность прогноза** зависит от:
                - Объема исторических данных
                - Волатильности валюты
                - Длительности прогноза
            
                > 📊 Чем короче период прогноза, тем он точнее
                """)
        
            with predict_col:
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
                    predict_button = st.button("🔮 Сделать прогноз", type="primary")
            
                if predict_button:
                    with st.spinner('Создание прогноза...'):
                        prediction_result = predict_currency(selected_code, days_to_predict)
                    
                        # Сохраняем прогноз в истории
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                    
                        # Добавляем текущий прогноз в историю
                        if prediction_result:
                            st.session_state.prediction_history.append({
                                'timestamp': datetime.now(),
                                'currency': selected_code,
                                'days': days_to_predict,
                                'result': prediction_result
                            })
                    
                        if prediction_result:
                            history = get_currency_history(selected_code, 30)
                        
                            # Создаем DataFrame с прогнозами
                            predictions = pd.DataFrame([
                                {"date": pd.to_datetime(p["date"]), "predicted_value": p["predicted_value"]}
                                for p in prediction_result["predictions"]
                            ])
                        
                            # Интерактивная визуализация прогноза
                            st.subheader("Прогноз курса")
                        
                            # Вкладки для разных представлений
                            viz_tabs = st.tabs(["График", "Таблица", "Метрики модели"])
                        
                            with viz_tabs[0]:
                                # Улучшенный график
                                fig = go.Figure()
                            
                                # Добавляем историю
                                fig.add_trace(go.Scatter(
                                    x=history['date'],
                                    y=history['value'],
                                    mode='lines',
                                    name='Исторические данные',
                                    line=dict(color='blue', width=2)
                                ))
                            
                                # Добавляем прогноз
                                fig.add_trace(go.Scatter(
                                    x=predictions['date'],
                                    y=predictions['predicted_value'],
                                    mode='lines+markers',
                                    name='Прогноз',
                                    line=dict(color='green', dash='dash', width=2),
                                    marker=dict(size=8, symbol='circle', color='green')
                                ))
                            
                                # Добавляем диапазон неопределенности
                                # Предполагаем 5% погрешность для иллюстрации
                                upper_bound = predictions['predicted_value'] * 1.05
                                lower_bound = predictions['predicted_value'] * 0.95
                            
                                # Заливка для диапазона неопределенности
                                fig.add_trace(go.Scatter(
                                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(0,176,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    hoverinfo="skip",
                                    showlegend=False
                                ))
                            
                                # Текущее значение
                                if not history.empty:
                                    current_value = history['value'].iloc[-1]
                                    fig.add_hline(y=current_value, line_dash="dot", 
                                                  line_color="red", annotation_text="Текущий курс")
                            
                                # Улучшенное форматирование
                                fig.update_layout(
                                    title=f"Прогноз курса {selected_code} на {days_to_predict} дней",
                                    xaxis_title="Дата",
                                    yaxis_title="Курс к рублю",
                                    height=500,
                                    hovermode="x unified",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="center",
                                        x=0.5
                                    ),
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                            
                                # Добавляем аннотацию
                                st.caption("Зеленая область показывает возможный диапазон колебаний курса")
                        
                            with viz_tabs[1]:
                                # Таблица с прогнозом
                                st.subheader("Детали прогноза")
                            
                                # Добавляем текущий курс для сравнения
                                current_value = history['value'].iloc[-1] if not history.empty else None
                            
                                # Преобразуем данные для отображения
                                df_display = pd.DataFrame([
                                    {
                                        "Дата": pd.to_datetime(p["date"]).strftime("%d.%m.%Y"),
                                        "День недели": pd.to_datetime(p["date"]).strftime("%a"),
                                        "Прогноз курса": f"{p['predicted_value']:.4f} ₽",
                                        "Изменение": f"{p['change_percent']:+.2f}%",
                                        "Изменение (абс.)": f"{(p['predicted_value'] - current_value):+.4f} ₽" if current_value else "N/A"
                                    }
                                    for p in prediction_result["predictions"]
                                ])
                            
                                # Стилизованная таблица
                                st.dataframe(
                                    df_display,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Изменение": st.column_config.ProgressColumn(
                                            "Изменение",
                                            format="%+.2f%%",
                                            min_value=-5,
                                            max_value=5,
                                        )
                                    }
                                )
                        
                            with viz_tabs[2]:
                                # Оценка качества модели
                                st.subheader("Оценка качества модели")
                            
                                # Используем метрики модели, полученные от API
                                if "model_metrics" in prediction_result:
                                    metrics = prediction_result["model_metrics"]
                                    model_type = prediction_result.get("model_type", "Неизвестная модель")
                                
                                    # Определяем качество модели
                                    r2_value = metrics.get('r2', 0)
                                    quality_class, quality_text = evaluate_model_quality(r2_value)
                                
                                    # Отображаем карточку с информацией о модели
                                    st.markdown(f"""
                                    <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:15px">
                                        <h4>Информация о модели</h4>
                                        <p><b>Тип модели:</b> {model_type}</p>
                                        <p><b>Версия:</b> {prediction_result['model_version']}</p>
                                        <p><b>Качество модели:</b> <span style="background-color:{
                                        '#d1fae5' if quality_class == 'good' else '#fef3c7' if quality_class == 'medium' else '#fee2e2'
                                        }; padding:3px 8px; border-radius:5px;">{quality_text}</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                    # Интерактивная визуализация метрик
                                    col1, col2 = st.columns(2)
                                
                                    with col1:
                                        # Диаграмма R²
                                        r2_chart = {
                                            'values': [r2_value, 1 - r2_value],
                                            'colors': ['#10b981', '#e5e7eb'],
                                            'title': 'R² (коэффициент детерминации)',
                                            'value': f"{r2_value:.4f}"
                                        }
                                    
                                        fig = go.Figure(go.Pie(
                                            values=r2_chart['values'],
                                            hole=0.7,
                                            marker_colors=r2_chart['colors'],
                                            showlegend=False,
                                            textinfo='none'
                                        ))
                                    
                                        fig.update_layout(
                                            title=r2_chart['title'],
                                            annotations=[dict(
                                                text=r2_chart['value'],
                                                x=0.5, y=0.5,
                                                font_size=20,
                                                showarrow=False
                                            )],
                                            height=250,
                                            margin=dict(t=30, b=0, l=0, r=0)
                                        )
                                    
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                    with col2:
                                        # Радиальная диаграмма для остальных метрик
                                        other_metrics = {
                                            'RMSE': metrics.get('rmse', 0),
                                            'MAE': metrics.get('mae', 0),
                                            'MAPE (%)': min(metrics.get('mape', 0), 100)  # Ограничиваем для визуализации
                                        }
                                    
                                        # Создаем нормализованные значения для визуализации
                                        values = list(other_metrics.values())
                                        max_value = max(values)
                                        norm_values = [v / max_value for v in values]
                                    
                                        fig = go.Figure()
                                    
                                        fig.add_trace(go.Scatterpolar(
                                            r=norm_values,
                                            theta=list(other_metrics.keys()),
                                            fill='toself',
                                            fillcolor='rgba(16, 185, 129, 0.2)',
                                            line=dict(color='#10b981')
                                        ))
                                    
                                        fig.update_layout(
                                            title="Другие метрики (относительные значения)",
                                            polar=dict(
                                                radialaxis=dict(
                                                    visible=True,
                                                    range=[0, 1]
                                                )
                                            ),
                                            showlegend=False,
                                            height=250,
                                            margin=dict(t=30, b=0, l=0, r=0)
                                        )
                                    
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                    # Таблица с метриками
                                    st.markdown("""
                                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                                        <tr style="background-color: #f8f9fa;">
                                            <th style="text-align: left; padding: 12px; border-bottom: 1px solid #e5e7eb;">Метрика</th>
                                            <th style="text-align: center; padding: 12px; border-bottom: 1px solid #e5e7eb;">Значение</th>
                                            <th style="text-align: left; padding: 12px; border-bottom: 1px solid #e5e7eb;">Описание</th>
                                        </tr>
                                        <tr>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">R² (коэффициент детерминации)</td>
                                            <td style="text-align: center; padding: 12px; border-bottom: 1px solid #e5e7eb;">{:.4f}</td>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">Доля объясненной дисперсии (0-1, чем ближе к 1, тем лучше)</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">MAE (средняя абсолютная ошибка)</td>
                                            <td style="text-align: center; padding: 12px; border-bottom: 1px solid #e5e7eb;">{:.4f}</td>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">Среднее абсолютных значений ошибок (чем меньше, тем лучше)</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">RMSE (корень из среднеквадратичной ошибки)</td>
                                            <td style="text-align: center; padding: 12px; border-bottom: 1px solid #e5e7eb;">{:.4f}</td>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">Мера типичного размера ошибки (чем меньше, тем лучше)</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">MAPE (средняя абсолютная процентная ошибка)</td>
                                            <td style="text-align: center; padding: 12px; border-bottom: 1px solid #e5e7eb;">{:.2f}%</td>
                                            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">Средняя ошибка в процентах (чем меньше, тем лучше)</td>
                                        </tr>
                                    </table>
                                    """.format(
                                        metrics.get('r2', 0),
                                        metrics.get('mae', 0),
                                        metrics.get('rmse', 0),
                                        metrics.get('mape', 0)
                                    ), unsafe_allow_html=True)
                                else:
                                    st.warning("Информация о метриках модели недоступна")
                        else:
                            st.error("Не удалось сделать прогноз")
        
            # История прогнозов
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                st.markdown("---")
                st.subheader("История ваших прогнозов")
            
                # Создаем таблицу с историей
                history_items = []
                for idx, pred in enumerate(st.session_state.prediction_history[::-1]):  # Обратный порядок для новых сверху
                    timestamp = pred['timestamp'].strftime("%d.%m.%Y %H:%M")
                    currency = pred['currency']
                    days = pred['days']
                
                    # Получаем первое и последнее значение из прогноза
                    if 'result' in pred and 'predictions' in pred['result'] and pred['result']['predictions']:
                        predictions = pred['result']['predictions']
                        first_value = predictions[0]['predicted_value']
                        last_value = predictions[-1]['predicted_value']
                        change = ((last_value / first_value) - 1) * 100
                        change_text = f"{change:+.2f}%"
                    else:
                        first_value = "N/A"
                        last_value = "N/A"
                        change_text = "N/A"
                
                    history_items.append({
                        "№": idx + 1,
                        "Дата": timestamp,
                        "Валюта": currency,
                        "Период": f"{days} дней",
                        "Изменение": change_text,
                        "Действия": "Просмотр"
                    })
            
                # Преобразуем в DataFrame для отображения
                if history_items:
                    df_history = pd.DataFrame(history_items)
                
                    # Используем вариант с выбором строки для просмотра деталей
                    selected_rows = st.dataframe(
                        df_history,
                        use_container_width=True,
                        hide_index=True,
                        selection="single"
                    )
                
                    # Если выбрана строка, показываем детали прогноза
                    if selected_rows:
                        selected_idx = list(selected_rows)[0]
                        if 0 <= selected_idx < len(st.session_state.prediction_history):
                            # Получаем выбранный прогноз
                            selected_pred = st.session_state.prediction_history[-(selected_idx+1)]  # Обратный индекс
                        
                            st.subheader(f"Детали прогноза от {selected_pred['timestamp'].strftime('%d.%m.%Y %H:%M')}")
                        
                            # Получаем данные прогноза
                            if 'result' in selected_pred and 'predictions' in selected_pred['result']:
                                # Создаем график
                                pred_df = pd.DataFrame([
                                    {"date": pd.to_datetime(p["date"]), "predicted_value": p["predicted_value"]}
                                    for p in selected_pred['result']["predictions"]
                                ])
                            
                                fig = go.Figure()
                            
                                # Добавляем прогноз
                                fig.add_trace(go.Scatter(
                                    x=pred_df['date'],
                                    y=pred_df['predicted_value'],
                                    mode='lines+markers',
                                    name='Прогноз',
                                    line=dict(color='green', width=2),
                                    marker=dict(size=8, symbol='circle', color='green')
                                ))
                            
                                fig.update_layout(
                                    title=f"Прогноз курса {selected_pred['currency']} на {selected_pred['days']} дней",
                                    xaxis_title="Дата",
                                    yaxis_title="Курс к рублю",
                                    height=300,
                                    hovermode="x unified"
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                            
                                # Отображаем таблицу с данными
                                st.dataframe(
                                    pd.DataFrame([
                                        {
                                            "Дата": pd.to_datetime(p["date"]).strftime("%d.%m.%Y"),
                                            "Прогноз курса": f"{p['predicted_value']:.4f} ₽",
                                            "Изменение": f"{p['change_percent']:+.2f}%"
                                        }
                                        for p in selected_pred['result']["predictions"]
                                    ]),
                                    use_container_width=True,
                                    hide_index=True
                                )
        
                    # Кнопка для очистки истории
                    if st.button("🗑️ Очистить историю прогнозов", key="clear_history"):
                        st.session_state.prediction_history = []
                        st.experimental_rerun()
        else:
            st.warning("Нет данных о валютах. Загрузите данные через боковую панель.")
    
    # ВКЛАДКА 4: Прогноз погоды
    with tab4:
        st.header("Прогноз погоды")
    
        # Добавляем информационную карточку
        st.info("""
        📌 **Для более точных прогнозов погоды рекомендуется:**
        - Использовать названия крупных городов
        - Выбирать период не более 30 дней для исторических данных
        - Включать ML-прогноз для сравнительного анализа
        """)
    
        # Получаем список городов
        popular_cities = get_popular_cities()
        city_names = [city["name"] for city in popular_cities]
    
        # Создаем две колонки: для выбора города и параметров
        city_col, params_col = st.columns([3, 2])
    
        with city_col:
            # Создаем вкладки для разных способов выбора города
            city_tabs = st.tabs(["Популярные города", "Поиск города", "Карта"])
        
            with city_tabs[0]:
                # Группируем города по странам
                countries = {}
                for city in popular_cities:
                    country = city["country"]
                    if country not in countries:
                        countries[country] = []
                    countries[country].append(city["name"])
            
                # Создаем кнопки для каждой страны
                selected_country = st.radio(
                    "Выберите страну:",
                    options=list(countries.keys()),
                    format_func=lambda c: {
                        "RU": "Россия 🇷🇺", 
                        "US": "США 🇺🇸", 
                        "GB": "Великобритания 🇬🇧",
                        "FR": "Франция 🇫🇷",
                        "DE": "Германия 🇩🇪",
                        "IT": "Италия 🇮🇹",
                        "JP": "Япония 🇯🇵",
                        "CN": "Китай 🇨🇳",
                        "AU": "Австралия 🇦🇺",
                        "AE": "ОАЭ 🇦🇪",
                        "TR": "Турция 🇹🇷"
                    }.get(c, c),
                    horizontal=True
                )
            
                # Показываем города для выбранной страны
                country_cities = countries.get(selected_country, [])
                cols = st.columns(2)
                city_buttons = []
            
                for i, city_name in enumerate(country_cities):
                    col_idx = i % 2
                    with cols[col_idx]:
                        city_btn = st.button(city_name, key=f"city_btn_{city_name}", use_container_width=True)
                        city_buttons.append((city_name, city_btn))
            
                # Проверяем, был ли выбран город
                selected_city = None
                for city_name, clicked in city_buttons:
                    if clicked:
                        selected_city = city_name
                        break
        
            with city_tabs[1]:
                # Поиск города
                search_city = st.text_input("🔍 Введите название города:", placeholder="Например: Москва, Париж, Нью-Йорк")
                search_button = st.button("Найти", key="search_city_btn")
            
                if search_button and search_city:
                    selected_city = search_city
        
            with city_tabs[2]:
                # Примечание: для настоящей карты потребуется интеграция с картами
                st.info("Функция выбора города на карте находится в разработке")
                st.image("https://via.placeholder.com/500x300.png?text=Interactive+Map+Coming+Soon", use_column_width=True)
            
                map_city_btn = st.button("Использовать выбранный на карте город")
                if map_city_btn:
                    selected_city = "Москва"  # Placeholder, в реальности здесь был бы город с карты
    
        with params_col:
            st.subheader("Параметры прогноза")
        
            # Параметры
            days_history = st.slider(
                "Дней для анализа:",
                min_value=7,
                max_value=60,
                value=30
            )
        
            # Опции для прогноза
            predict = st.checkbox("Использовать ML-прогноз", value=True)
        
            # Дополнительные опции вывода
            show_historical = st.checkbox("Показать исторические данные", value=True)
            show_details = st.checkbox("Детальная статистика", value=False)
    
        # Объединяем выбор города из разных источников
        city = selected_city if selected_city else ""
    
        # Кнопка запроса прогноза погоды
        get_weather_btn = st.button("🌤️ Получить прогноз погоды", type="primary", use_container_width=True)
    
        if city and get_weather_btn:
            with st.spinner(f"Получение данных о погоде для {city}..."):
                weather_data = get_weather(city, days_history, predict)
        
            if weather_data:
                # Создаем карточку с текущей погодой
                current = weather_data['current']
                current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
            
                # Карточка с текущей погодой
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin: 0;">{weather_data['city']}</h2>
                            <p style="color: #6c757d; margin-top: 5px;">{current_date}</p>
                        </div>
                        <div style="text-align: center;">
                            <img src="http://openweathermap.org/img/wn/{current['icon']}@2x.png" alt="{current['description']}" style="width: 100px;">
                            <p style="margin: 0;">{current['description'].capitalize()}</p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 3rem; font-weight: bold;">{current['temperature']:.1f}°C</span>
                            <p style="margin: 0;">Ощущается как {current['feels_like']:.1f}°C</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
                # Основные метрики
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.metric("Влажность", f"{current['humidity']}%")
            
                with col2:
                    st.metric("Давление", f"{current['pressure']} гПа")
            
                with col3:
                    st.metric("Ветер", f"{current['wind_speed']} м/с")
            
                # Вкладки с прогнозом
                forecast_tabs = st.tabs(["Прогноз на неделю", "Почасовой прогноз", "Сравнение прогнозов", "Исторические данные"])
            
                with forecast_tabs[0]:
                    # Недельный прогноз в виде карточек
                    st.subheader("Прогноз на 5 дней")
                
                    # Создаем колонки для дней
                    day_cols = st.columns(min(5, len(weather_data['forecast'])))
                
                    for i, day in enumerate(weather_data['forecast'][:5]):
                        # Преобразуем дату
                        date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
                        date_formatted = date_obj.strftime("%d.%m")
                        day_of_week = translate_day_of_week(date_obj.strftime("%A"))
                    
                        with day_cols[i]:
                            st.markdown(f"""
                            <div style="background-color: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <p style="font-weight: bold; margin-bottom: 5px;">{day_of_week}, {date_formatted}</p>
                                <img src="http://openweathermap.org/img/wn/{day['icon']}@2x.png" alt="{day['description']}" style="width: 50px;">
                                <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{day['temperature']:.1f}°C</p>
                                <p style="font-size: 0.8rem; color: #6c757d; margin-top: 0;">Ощущается: {day['feels_like']:.1f}°C</p>
                                <p style="margin-bottom: 0;">{day['description'].capitalize()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                    # График прогноза температуры
                    st.subheader("График температуры")
                
                    # Создаем DataFrame для графика
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
                    st.info("Почасовой прогноз находится в разработке")
                
                    # Здесь в реальном приложении можно добавить почасовой прогноз,
                    # если API предоставляет такие данные
            
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
                            use_container_width=True,
                            hide_index=True
                        )
                    
                        # Диаграмма сравнения
                        st.subheader("Диаграмма разницы прогнозов")
                    
                        # Данные для графика разницы
                        comparison_df = comparison_df.dropna(subset=["Разница"])
                        if not comparison_df.empty:
                            dates_fmt = comparison_df["Дата"].dt.strftime("%d.%m")
                        
                            fig = go.Figure()
                        
                            fig.add_trace(go.Bar(
                                x=dates_fmt,
                                y=comparison_df["Разница"],
                                marker_color=['green' if d >= 0 else 'red' for d in comparison_df["Разница"]],
                                text=[f"{d:+.1f}°C" for d in comparison_df["Разница"]],
                                textposition='auto'
                            ))
                        
                            fig.update_layout(
                                title="Разница между ML-моделью и OpenWeatherMap (ML минус OWM)",
                                xaxis_title="Дата",
                                yaxis_title="Разница, °C",
                                height=350
                            )
                        
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("ML-прогноз недоступен. Включите опцию 'ML-прогноз'")
            
                with forecast_tabs[3]:
                    # Исторические данные
                    if show_historical and 'historical_data' in weather_data and weather_data['historical_data']:
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
                            title=f"Историческая температура в {weather_data['city']}",
                            markers=True
                        )
                    
                        fig.update_layout(
                            xaxis_title="Дата",
                            yaxis_title="Температура, °C",
                            height=350
                        )
                    
                        st.plotly_chart(fig, use_container_width=True)
                    
                        if show_details:
                            # Сводная статистика
                            st.subheader("Сводная статистика")
                        
                            col1, col2, col3, col4 = st.columns(4)
                        
                            with col1:
                                st.metric("Средняя", f"{hist_data['Температура'].mean():.1f}°C")
                        
                            with col2:
                                st.metric("Минимум", f"{hist_data['Температура'].min():.1f}°C")
                        
                            with col3:
                                st.metric("Максимум", f"{hist_data['Температура'].max():.1f}°C")
                        
                            with col4:
                                temp_range = hist_data['Температура'].max() - hist_data['Температура'].min()
                                st.metric("Размах", f"{temp_range:.1f}°C")
                        
                            # Таблица с историческими данными
                            with st.expander("Просмотреть подробные исторические данные"):
                                # Форматируем дату для отображения
                                display_hist = hist_data.copy()
                                display_hist["Дата"] = display_hist["Дата"].dt.strftime("%d.%m.%Y")
                            
                                st.dataframe(
                                    display_hist,
                                    use_container_width=True,
                                    hide_index=True
                                )
                    else:
                        st.info("Исторические данные скрыты. Включите опцию 'Показать исторические данные'.")
            else:
                st.error(f"Не удалось получить данные о погоде для '{city}'")
        else:
            st.info("Выберите город и нажмите 'Получить прогноз'")
    
    # ВКЛАДКА 5: О проекте
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