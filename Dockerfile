FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for nmap
RUN apt-get update && apt-get install -y nmap && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app
ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:create_app()"]