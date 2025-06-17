#!/usr/bin/env python3
"""
تطبيق البحث الدلالي باستخدام FastAPI
يستخدم النموذج المدرب المحفوظ في Docker container
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
import uvicorn

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from semantic_search import SemanticSearchEngine

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# إنشاء مجلد السجلات
os.makedirs('/app/logs', exist_ok=True)

# تكوين المسارات
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/fine_tuned_model")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "/app/embeddings")
DATA_PATH = os.getenv("DATA_PATH", "/app/data")

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="نظام البحث الدلالي",
    description="نظام بحث ذكي باستخدام الذكاء الاصطناعي",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# إعداد القوالب والملفات الثابتة
templates = Jinja2Templates(directory="templates")

# متغير عام لمحرك البحث
search_engine: Optional[SemanticSearchEngine] = None

# نماذج البيانات
class SearchRequest(BaseModel):
    query: str
    top_k: int = 15

class SearchResult(BaseModel):
    rank: int
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """تهيئة التطبيق عند البدء"""
    global search_engine
    
    try:
        logger.info("🚀 بدء تهيئة نظام البحث الدلالي...")
        
        # إنشاء محرك البحث
        search_engine = SemanticSearchEngine(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            embeddings_path=EMBEDDINGS_PATH
        )
        
        # تحميل النموذج والبيانات
        await search_engine.initialize()
        
        logger.info("✅ تم تهيئة نظام البحث بنجاح!")
        
    except Exception as e:
        logger.error(f"❌ فشل في تهيئة النظام: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """تنظيف الموارد عند الإغلاق"""
    logger.info("🔄 إغلاق النظام...")

@app.get("/health")
async def health_check():
    """فحص صحة التطبيق"""
    if search_engine and search_engine.is_ready():
        return {
            "status": "healthy",
            "message": "نظام البحث يعمل بشكل طبيعي",
            "model_loaded": True
        }
    else:
        raise HTTPException(status_code=503, detail="النظام غير جاهز")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """الصفحة الرئيسية"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "نظام البحث الدلالي"}
    )

@app.post("/search", response_model=SearchResponse)
async def search_api(search_request: SearchRequest):
    """API للبحث الدلالي"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="محرك البحث غير متاح")
    
    if not search_request.query.strip():
        raise HTTPException(status_code=400, detail="يجب كتابة سؤال للبحث")
    
    try:
        import time
        start_time = time.time()
        
        # تنفيذ البحث
        results = await search_engine.search(
            query=search_request.query,
            top_k=search_request.top_k
        )
        
        processing_time = time.time() - start_time
        
        # تحويل النتائج للصيغة المطلوبة
        search_results = [
            SearchResult(
                rank=result["rank"],
                id=str(result["id"]),
                text=result["text"],
                score=float(result["score"])
            )
            for result in results
        ]
        
        return SearchResponse(
            query=search_request.query,
            results=search_results,
            total_found=len(search_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"خطأ في البحث: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في البحث: {str(e)}")

@app.post("/search-form")
async def search_form(request: Request, query: str = Form(...), top_k: int = Form(15)):
    """البحث عبر النموذج HTML"""
    try:
        search_request = SearchRequest(query=query, top_k=top_k)
        response = await search_api(search_request)
        
        return templates.TemplateResponse(
            "results.html", 
            {
                "request": request,
                "query": query,
                "results": response.results,
                "total_found": response.total_found,
                "processing_time": response.processing_time,
                "title": "نتائج البحث"
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": e.detail,
                "title": "خطأ في البحث"
            }
        )

@app.get("/stats")
async def get_stats():
    """إحصائيات النظام"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="محرك البحث غير متاح")
    
    try:
        stats = await search_engine.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"خطأ في الحصول على الإحصائيات: {str(e)}")
        raise HTTPException(status_code=500, detail="خطأ في الحصول على الإحصائيات")

@app.get("/model-info")
async def get_model_info():
    """معلومات النموذج"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="محرك البحث غير متاح")
    
    try:
        info = await search_engine.get_model_info()
        return info
    except Exception as e:
        logger.error(f"خطأ في الحصول على معلومات النموذج: {str(e)}")
        raise HTTPException(status_code=500, detail="خطأ في الحصول على معلومات النموذج")

if __name__ == "__main__":
    # تشغيل التطبيق محلياً للتطوير
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )