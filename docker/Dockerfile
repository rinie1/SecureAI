# Dockerfile for SecureAI server
FROM python:3.13-slim

RUN useradd --create-home appuser

WORKDIR /app
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../server.py ../mnist_model_split.npz ../model_params.json ./

USER appuser

EXPOSE 8000
CMD ["python", "server.py"]
