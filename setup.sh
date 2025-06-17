#!/bin/bash

# Script to setup and run the Semantic Search Docker application
# نص لإعداد وتشغيل تطبيق البحث الدلالي

echo "🚀 إعداد تطبيق البحث الدلالي"
echo "=================================="

# التحقق من وجود Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker غير مثبت. يرجى تثبيت Docker أولاً"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose غير مثبت. يرجى تثبيت Docker Compose أولاً"
    exit 1
fi

echo "✅ Docker و Docker Compose متوفران"

# التحقق من وجود الملفات المطلوبة
echo "🔍 التحقق من الملفات المطلوبة..."

REQUIRED_FILES=(
    "models/fine_tuned_model/config.json"
    "models/fine_tuned_model/model.safetensors"
    "models/fine_tuned_model/tokenizer.json"
    "app/main.py"
    "app/requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
)

MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ ملفات مطلوبة مفقودة:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "يرجى التأكد من وجود جميع الملفات المطلوبة قبل التشغيل"
    exit 1
fi

echo "✅ جميع الملفات المطلوبة موجودة"

# إنشاء المجلدات المطلوبة
echo "📁 إنشاء المجلدات المطلوبة..."
mkdir -p logs
mkdir -p app/templates

# إنشاء ملف templates إذا لم يكن موجوداً
if [ ! -d "app/templates" ]; then
    mkdir -p app/templates
    echo "⚠️ يرجى إضافة ملفات HTML في مجلد app/templates/"
fi

# التحقق من حجم النموذج
MODEL_SIZE=$(du -h models/fine_tuned_model/model.safetensors 2>/dev/null | cut -f1)
if [ ! -z "$MODEL_SIZE" ]; then
    echo "📊 حجم النموذج: $MODEL_SIZE"
fi

# بناء الحاوية
echo "🔨 بناء حاوية Docker..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ فشل في بناء الحاوية"
    exit 1
fi

echo "✅ تم بناء الحاوية بنجاح"

# تشغيل الخدمة
echo "🚀 تشغيل التطبيق..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ فشل في تشغيل التطبيق"
    exit 1
fi

echo ""
echo "🎉 تم تشغيل التطبيق بنجاح!"
echo "=================================="
echo "🌐 رابط التطبيق: http://localhost:8000"
echo "📚 وثائق API: http://localhost:8000/docs"
echo "📊 الإحصائيات: http://localhost:8000/stats"
echo ""
echo "📋 أوامر مفيدة:"
echo "   عرض السجلات: docker-compose logs -f"
echo "   إيقاف التطبيق: docker-compose down"
echo "   إعادة تشغيل: docker-compose restart"
echo "   دخول الحاوية: docker-compose exec semantic-search bash"
echo ""

# انتظار تشغيل الخدمة
echo "⏳ انتظار تشغيل الخدمة..."
sleep 10

# التحقق من حالة الخدمة
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ الخدمة تعمل بشكل صحيح!"
    echo "🎯 يمكنك الآن فتح المتصفح والذهاب إلى: http://localhost:8000"
else
    echo "⚠️ الخدمة قد تحتاج وقت إضافي للتشغيل"
    echo "💡 استخدم 'docker-compose logs -f' لمراقبة السجلات"
fi

echo ""
echo "🔧 للتطوير:"
echo "   تعديل الكود: قم بتعديل الملفات في مجلد app/"
echo "   إعادة البناء: docker-compose build && docker-compose up -d"
echo ""
echo "Happy searching! 🎉"