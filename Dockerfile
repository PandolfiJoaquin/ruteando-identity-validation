# Docker file that installs all dependencies
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

