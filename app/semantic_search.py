"""
محرك البحث الدلالي الأساسي
يدير تحميل النموذج والبيانات وتنفيذ البحث
"""

import os
import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """محرك البحث الدلالي الرئيسي"""
    
    def __init__(self, model_path: str, data_path: str, embeddings_path: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        
        # متغيرات النظام
        self.model: Optional[SentenceTransformer] = None
        self.knowledge_base_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self._is_ready = False
        
        # إحصائيات
        self.search_count = 0
        self.total_processing_time = 0.0
        
    async def initialize(self):
        """تهيئة النظام بالكامل"""
        try:
            logger.info("🔄 بدء تهيئة محرك البحث...")
            
            # تحميل النموذج
            await self._load_model()
            
            # تحميل قاعدة المعرفة
            await self._load_knowledge_base()
            
            # تحميل الـ embeddings
            await self._load_embeddings()
            
            self._is_ready = True
            logger.info("✅ تم تهيئة محرك البحث بنجاح!")
            
        except Exception as e:
            logger.error(f"❌ فشل في تهيئة محرك البحث: {str(e)}")
            raise
    
    async def _load_model(self):
        """تحميل النموذج المدرب"""
        try:
            logger.info(f"📥 تحميل النموذج من: {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"مسار النموذج غير موجود: {self.model_path}")
            
            # تحقق من وجود الملفات الأساسية
            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "sentence_bert_config.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.model_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"ملفات مطلوبة مفقودة: {missing_files}")
            
            # تحميل النموذج
            self.model = SentenceTransformer(str(self.model_path))
            logger.info("✅ تم تحميل النموذج بنجاح!")
            
        except Exception as e:
            logger.error(f"❌ فشل في تحميل النموذج: {str(e)}")
            # محاولة تحميل النموذج الأساسي كبديل
            try:
                logger.info("🔄 محاولة تحميل النموذج الأساسي...")
                self.model = SentenceTransformer("intfloat/multilingual-e5-large")
                logger.warning("⚠️ تم تحميل النموذج الأساسي بدلاً من المدرب")
            except Exception as fallback_error:
                logger.error(f"❌ فشل في تحميل النموذج الأساسي: {str(fallback_error)}")
                raise
    
    async def _load_knowledge_base(self):
        """تحميل قاعدة المعرفة"""
        try:
            # البحث عن ملف قاعدة البيانات
            data_files = list(self.data_path.glob("*.xlsx"))
            if not data_files:
                raise FileNotFoundError(f"لم يتم العثور على ملفات Excel في: {self.data_path}")
            
            # استخدام أول ملف موجود
            data_file = data_files[0]
            logger.info(f"📚 تحميل قاعدة المعرفة من: {data_file}")
            
            # تحميل البيانات
            self.knowledge_base_df = pd.read_excel(data_file, dtype=str)
            
            # التحقق من الأعمدة المطلوبة
            required_columns = ["id", "text"]
            missing_columns = [col for col in required_columns if col not in self.knowledge_base_df.columns]
            
            if missing_columns:
                raise ValueError(f"أعمدة مطلوبة مفقودة: {missing_columns}")
            
            # تنظيف البيانات
            original_count = len(self.knowledge_base_df)
            self.knowledge_base_df = self.knowledge_base_df.dropna(subset=["text"])
            self.knowledge_base_df["text"] = self.knowledge_base_df["text"].str.strip()
            self.knowledge_base_df = self.knowledge_base_df[self.knowledge_base_df["text"] != ""]
            
            final_count = len(self.knowledge_base_df)
            logger.info(f"✅ تم تحميل قاعدة المعرفة: {final_count} سجل (من أصل {original_count})")
            
        except Exception as e:
            logger.error(f"❌ فشل في تحميل قاعدة المعرفة: {str(e)}")
            raise
    
    async def _load_embeddings(self):
        """تحميل أو توليد الـ embeddings"""
        try:
            # البحث عن ملف embeddings
            embedding_files = list(self.embeddings_path.glob("*.npy"))
            
            if embedding_files:
                # تحميل embeddings موجودة
                embedding_file = embedding_files[0]
                logger.info(f"📥 تحميل embeddings من: {embedding_file}")
                self.embeddings = np.load(embedding_file)
                
                # التحقق من تطابق الأبعاد
                if len(self.embeddings) != len(self.knowledge_base_df):
                    logger.warning("⚠️ عدم تطابق أبعاد embeddings مع قاعدة البيانات، سيتم إعادة التوليد")
                    await self._generate_embeddings()
                else:
                    logger.info(f"✅ تم تحميل embeddings: {self.embeddings.shape}")
            else:
                # توليد embeddings جديدة
                logger.info("🔄 لم يتم العثور على embeddings، سيتم التوليد...")
                await self._generate_embeddings()
                
        except Exception as e:
            logger.error(f"❌ فشل في تحميل embeddings: {str(e)}")
            # محاولة توليد embeddings جديدة
            await self._generate_embeddings()
    
    async def _generate_embeddings(self):
        """توليد embeddings جديدة"""
        try:
            logger.info("🔄 بدء توليد embeddings...")
            
            texts = self.knowledge_base_df["text"].tolist()
            
            # توليد embeddings على دفعات
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"🔄 معالجة الدفعة {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                ).cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
                
                # إعطاء فرصة للمهام الأخرى
                await asyncio.sleep(0.01)
            
            # دمج جميع embeddings
            self.embeddings = np.vstack(all_embeddings)
            
            # حفظ embeddings
            output_file = self.embeddings_path / "generated_embeddings.npy"
            np.save(output_file, self.embeddings)
            
            logger.info(f"✅ تم توليد وحفظ embeddings: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"❌ فشل في توليد embeddings: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """تنفيذ البحث الدلالي"""
        if not self._is_ready:
            raise RuntimeError("محرك البحث غير جاهز")
        
        if not query.strip():
            raise ValueError("الاستعلام فارغ")
        
        try:
            import time
            start_time = time.time()
            
            # تشفير الاستعلام
            query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
            
            # حساب التشابه
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # الحصول على أفضل النتائج
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # تجميع النتائج
            results = []
            for rank, idx in enumerate(top_indices, 1):
                record_id = self.knowledge_base_df.iloc[idx]["id"]
                record_text = self.knowledge_base_df.iloc[idx]["text"]
                score = float(similarities[idx])
                
                results.append({
                    "rank": rank,
                    "id": record_id,
                    "text": record_text,
                    "score": score
                })
            
            # تحديث الإحصائيات
            processing_time = time.time() - start_time
            self.search_count += 1
            self.total_processing_time += processing_time
            
            logger.info(f"🔍 تم البحث عن: '{query}' في {processing_time:.3f} ثانية")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ فشل في البحث: {str(e)}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام"""
        avg_processing_time = (
            self.total_processing_time / self.search_count 
            if self.search_count > 0 else 0
        )
        
        return {
            "total_searches": self.search_count,
            "average_processing_time": avg_processing_time,
            "knowledge_base_size": len(self.knowledge_base_df) if self.knowledge_base_df is not None else 0,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "model_loaded": self.model is not None,
            "system_ready": self._is_ready
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النموذج"""
        info = {
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None
        }
        
        # محاولة قراءة معلومات التدريب
        try:
            training_info_file = self.model_path / "training_info.json"
            if training_info_file.exists():
                import json
                with open(training_info_file, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                info["training_info"] = training_info
        except Exception as e:
            logger.warning(f"لم يتم العثور على معلومات التدريب: {str(e)}")
        
        return info
    
    def is_ready(self) -> bool:
        """التحقق من جاهزية النظام"""
        return self._is_ready