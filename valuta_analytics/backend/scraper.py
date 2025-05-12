
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time

class CurrencyScraper:
    def __init__(self):
        """Инициализация скрапера для ЦБ РФ"""
        self.base_url = "https://www.cbr-xml-daily.ru/daily_json.js"
        self.archive_url = "https://www.cbr-xml-daily.ru/archive/{date}/daily_json.js"
    
    def get_current_rates(self):
        """Получение текущих курсов валют"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            data = response.json()
            return self._parse_json_data(data)
        except Exception as e:
            print(f"Ошибка при получении текущих курсов: {e}")
            return []
    
    def get_historical_rates(self, start_date, end_date=None):
        """Получение исторических данных о курсах валют"""
        if end_date is None:
            end_date = datetime.now().date()
    
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
        # Проверка, что даты не в будущем
        current_date = datetime.now().date()
        if start_date > current_date:
            start_date = current_date
        if end_date > current_date:
            end_date = current_date
    
        all_data = []
        current_date = start_date
    
        while current_date <= end_date:
            date_str = current_date.strftime("%Y/%m/%d")
            url = self.archive_url.format(date=date_str)
        
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                all_data.extend(self._parse_json_data(data))
                print(f"Загружены данные за {current_date}")
            except Exception as e:
                print(f"Ошибка при получении данных за {current_date}: {e}")
        
            # Добавляем задержку, чтобы не нагружать сервер
            time.sleep(0.5)
            current_date += timedelta(days=1)
    
        return all_data
    
    def _parse_json_data(self, data):
        """Парсинг JSON данных из API ЦБ РФ"""
        currencies = []
        date = data.get('Date', '').split('T')[0]
        
        for code, currency_data in data.get('Valute', {}).items():
            currencies.append({
                'date': date,
                'currency_code': code,
                'nominal': currency_data.get('Nominal', 1),
                'name': currency_data.get('Name', ''),
                'value': currency_data.get('Value', 0.0)
            })
        
        return currencies