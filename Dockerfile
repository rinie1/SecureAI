FROM python:3.10-slim

# Set up a working directory inside the container
WORKDIR /app

# Copying dependencies
COPY /server/requirements.txt .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code of the application
COPY /server/ /app/

# Command to run the application
CMD ["python", "server.py"]
