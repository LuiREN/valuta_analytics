import pytest
import sys
import os

# Добавляем корневую директорию проекта в path
# Это позволит импортировать модули из корня проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
