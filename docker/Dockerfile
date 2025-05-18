# Dockerfile for SecureAI server
FROM python:3.13-slim

# Создаём непривилегированного пользователя
RUN useradd --create-home appuser

WORKDIR /app
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервера и модель
COPY ../server.py ../mnist_model_split.npz ../model_params.json ./

# Переключаемся на непользователя
USER appuser

EXPOSE 8000
CMD ["python", "server.py"]
