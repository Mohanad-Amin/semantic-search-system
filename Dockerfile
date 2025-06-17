FROM python:3.9-slim

WORKDIR /app

# تثبيت dependencies النظام
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# نسخ وتثبيت requirements
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ التطبيق
COPY app/ .

# إنشاء مجلدات
RUN mkdir -p data embeddings logs

# متغيرات البيئة
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "main.py"]
