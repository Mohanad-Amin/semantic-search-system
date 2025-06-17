
import os
import torch
from sentence_transformers import SentenceTransformer

class CPUOptimizedConfig:
    """إعدادات محسنة للتشغيل على CPU"""
    
    @staticmethod
    def setup_cpu_environment():
        """إعداد البيئة للتشغيل المحسن على CPU"""
        
        # تعطيل CUDA نهائياً
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # تحسين استخدام CPU threads
        # استخدام جميع النوى المتاحة
        cpu_count = os.cpu_count()
        torch.set_num_threads(cpu_count)
        
        # تحسين BLAS للعمليات الرياضية
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        
        # تحسين tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        print(f"✅ تم إعداد التشغيل على CPU مع {cpu_count} نواة")
    
    @staticmethod
    def get_optimal_batch_size():
        """حساب حجم الدفعة الأمثل حسب الذاكرة المتاحة"""
        try:
            import psutil
            
            # الحصول على الذاكرة المتاحة (GB)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            if available_memory >= 12:
                return 64  # ذاكرة كبيرة
            elif available_memory >= 8:
                return 32  # ذاكرة متوسطة
            elif available_memory >= 4:
                return 16  # ذاكرة قليلة
            else:
                return 8   # ذاكرة محدودة جداً
                
        except ImportError:
            # إذا لم تكن psutil متوفرة، استخدم قيمة افتراضية
            return 16
    
    @staticmethod
    def load_model_cpu_optimized(model_path: str):
        """تحميل النموذج مع تحسينات CPU"""
        
        # إعداد البيئة
        CPUOptimizedConfig.setup_cpu_environment()
        
        try:
            print("🔄 تحميل النموذج مع تحسينات CPU...")
            
            # تحميل النموذج مع إجبار استخدام CPU
            model = SentenceTransformer(
                model_path,
                device='cpu'  # إجبار استخدام CPU
            )
            
            # تحسينات إضافية
            model.eval()  # وضع التقييم لتحسين الأداء
            
            # تحسين دقة العمليات (اختياري)
            # model.half()  # تقليل دقة الأرقام لتوفير الذاكرة (قد يؤثر على الدقة)
            
            print("✅ تم تحميل النموذج بنجاح مع تحسينات CPU")
            return model
            
        except Exception as e:
            print(f"❌ خطأ في تحميل النموذج: {e}")
            raise
    
    @staticmethod
    def get_cpu_info():
        """الحصول على معلومات المعالج"""
        try:
            import psutil
            import platform
            
            cpu_info = {
                "processor": platform.processor(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "غير معروف",
                "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),
                "pytorch_threads": torch.get_num_threads()
            }
            
            return cpu_info
            
        except ImportError:
            return {"error": "psutil غير متوفر لعرض معلومات النظام"}


# تطبيق التحسينات في semantic_search.py
class CPUOptimizedSemanticSearch:
    """محرك بحث محسن للـ CPU"""
    
    def __init__(self, model_path: str, data_path: str, embeddings_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        
        # إعداد التحسينات
        self.cpu_config = CPUOptimizedConfig()
        self.batch_size = self.cpu_config.get_optimal_batch_size()
        
        print(f"🔧 حجم الدفعة المحسن: {self.batch_size}")
        
        # متغيرات النظام
        self.model = None
        self.knowledge_base_df = None
        self.embeddings = None
        self._is_ready = False
    
    async def _load_model_optimized(self):
        """تحميل النموذج مع تحسينات CPU"""
        try:
            self.model = self.cpu_config.load_model_cpu_optimized(self.model_path)
            
            # طباعة معلومات النظام
            cpu_info = self.cpu_config.get_cpu_info()
            print("💻 معلومات النظام:")
            for key, value in cpu_info.items():
                print(f"   {key}: {value}")
                
        except Exception as e:
            print(f"❌ فشل تحميل النموذج المحسن: {e}")
            # التراجع للطريقة العادية
            self.model = SentenceTransformer(self.model_path, device='cpu')
    
    async def _generate_embeddings_cpu_optimized(self):
        """توليد embeddings محسن للـ CPU"""
        try:
            print("🔄 بدء توليد embeddings مع تحسينات CPU...")
            
            texts = self.knowledge_base_df["text"].tolist()
            total_texts = len(texts)
            
            # استخدام دفعات صغيرة مع progress bar
            all_embeddings = []
            batch_size = self.batch_size
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                progress = (i + len(batch_texts)) / total_texts * 100
                
                print(f"🔄 معالجة: {progress:.1f}% ({i + len(batch_texts)}/{total_texts})")
                
                # توليد embeddings للدفعة
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,  # استخدام numpy مباشرة
                    device='cpu',
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True  # تطبيع للحصول على نتائج أفضل
                )
                
                all_embeddings.append(batch_embeddings)
                
                # إعطاء فرصة للنظام للتنفس
                await asyncio.sleep(0.1)
            
            # دمج جميع embeddings
            import numpy as np
            self.embeddings = np.vstack(all_embeddings)
            
            print(f"✅ تم توليد embeddings: {self.embeddings.shape}")
            
        except Exception as e:
            print(f"❌ فشل في توليد embeddings: {e}")
            raise