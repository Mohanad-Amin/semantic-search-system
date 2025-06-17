# استخدام Python 3.10 كقاعدة
FROM python:3.10-slim

# تعيين متغيرات البيئة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DISABLED=true \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=true \
    TOKENIZERS_PARALLELISM=false

# تعيين مجلد العمل
WORKDIR /app

# تثبيت متطلبات النظام
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف متطلبات Python
COPY app/requirements.txt .

# تثبيت مكتبات Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# إنشاء مستخدم غير جذر
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/data /app/embeddings && \
    chown -R appuser:appuser /app

# نسخ النموذج المدرب
COPY --chown=appuser:appuser models/fine_tuned_model/ /app/models/fine_tuned_model/

# نسخ ملفات البيانات والـ embeddings
COPY --chown=appuser:appuser embeddings/ /app/embeddings/
COPY --chown=appuser:appuser data/ /app/data/

# نسخ كود التطبيق
COPY --chown=appuser:appuser app/ /app/

# التبديل للمستخدم غير الجذر
USER appuser

# كشف المنفذ
EXPOSE 8000

# فحص صحة الحاوية
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]