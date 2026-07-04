"""
VoicePrint System - 主入口
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.engine import VoicePrintEngine
from app.core.store import VoicePrintStore
from app.api.routes import init_routes, router as api_router

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 配置
DATA_DIR = os.getenv("VP_DATA_DIR", "./data")
MODEL_DIR = os.getenv("VP_MODEL_DIR", "./models")
HOST = os.getenv("VP_HOST", "0.0.0.0")
PORT = int(os.getenv("VP_PORT", "8700"))

# 全局实例
engine = None
store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global engine, store
    
    logger.info("=" * 60)
    logger.info("VoicePrint Recognition System starting...")
    logger.info(f"  Data dir: {DATA_DIR}")
    logger.info(f"  Model dir: {MODEL_DIR}")
    logger.info(f"  Host: {HOST}:{PORT}")
    logger.info("=" * 60)
    
    # 初始化引擎和存储
    engine = VoicePrintEngine(model_cache_dir=MODEL_DIR)
    store = VoicePrintStore(db_path=os.path.join(DATA_DIR, "voiceprint.db"))
    
    # 注入到路由
    init_routes(engine, store)
    
    logger.info("System ready!")
    logger.info(f"API docs: http://localhost:{PORT}/docs")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="VoicePrint Recognition System",
    description="""
    ## 声纹识别系统
    
    基于 ECAPA-TDNN 的声纹识别，支持：
    
    - **声纹注册** - 上传音频注册用户声纹
    - **身份验证** - 1:1 说话人确认
    - **说话人识别** - 1:N 辨认
    - **说话人分离** - 多人对话中分离各说话人
    - **目标提取** - 从混合语音中只提取指定用户的声音
    
    适用于 AI 眼镜、桌面机器人等设备接入。
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - 允许所有来源（生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "name": "VoicePrint Recognition System",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "enroll": "POST /api/enroll",
            "verify": "POST /api/verify",
            "identify": "POST /api/identify",
            "diarize": "POST /api/diarize",
            "extract_target": "POST /api/extract_target",
            "speakers": "GET /api/speakers",
            "stats": "GET /api/stats",
            "health": "GET /api/health",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
