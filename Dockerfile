FROM python:3.9-slim

WORKDIR /app

# نسخ requirements أولاً للاستفادة من Docker cache
COPY app/requirements.txt .

# تثبيت dependencies
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود التطبيق
COPY app/ .

# إنشاء مجلدات فارغة للنماذج والبيانات
RUN mkdir -p models embeddings data logs

# تعريف متغير البيئة
ENV PYTHONPATH=/app

# فتح port
EXPOSE 8000

# تشغيل التطبيق
CMD ["python", "main.py"]
