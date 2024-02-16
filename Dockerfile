# Установка базового образа Python
FROM python:3.9

# Копирование файлов в Docker-контейнер
COPY app.py model.cbm input_sample.json /app/

# Установка зависимостей
RUN pip install catboost

# Установка рабочей директории
WORKDIR /app

# Запуск приложения
CMD ["python", "app.py"]
