FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# In development, source is mounted via docker-compose volume
# This copy is retained for standalone production builds
COPY . /app