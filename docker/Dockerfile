# Используем официальный образ Python в качестве базового
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости
COPY ../server/requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY ../server/ .

# Указываем порт, который будет прослушиваться
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "server.py", "--port 5000"]
