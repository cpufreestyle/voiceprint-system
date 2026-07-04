"""
VoicePrint API - FastAPI 路由
注册 / 验证 / 识别 / 分离 / 目标提取
"""

import io
import time
import tempfile
import numpy as np
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import logging

from app.core.engine import VoicePrintEngine, VerifyResult, IdentifyResult
from app.core.store import VoicePrintStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# 全局实例（由 main.py 初始化）
engine: VoicePrintEngine = None  # type: ignore
store: VoicePrintStore = None    # type: ignore


def init_routes(eng: VoicePrintEngine, st: VoicePrintStore):
    global engine, store
    engine = eng
    store = st


async def _load_upload(upload: UploadFile, target_sr: int = 16000) -> tuple:
    """从上传文件加载音频"""
    import soundfile as sf
    import librosa
    
    content = await upload.read()
    
    # 写临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 用 librosa 加载（自动重采样 + mono）
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    except Exception as e:
        logger.error(f"Audio load failed: {e}")
        # fallback: soundfile
        audio, sr = sf.read(tmp_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    finally:
        import os
        os.unlink(tmp_path)


@router.post("/enroll")
async def enroll(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    name: str = Form(...),
):
    """注册声纹"""
    try:
        audio_data, sr = await _load_upload(audio)
        
        if len(audio_data) < sr * 0.5:
            raise HTTPException(400, "Audio too short (min 0.5s)")
        
        # 提取声纹
        embedding = engine.extract_embedding(audio_data, sr)
        
        # 存库
        success = store.enroll(user_id, name, embedding)
        
        if not success:
            raise HTTPException(500, "Enroll failed")
        
        return {
            "success": True,
            "user_id": user_id,
            "name": name,
            "duration": float(len(audio_data) / sr),
            "embedding_dim": len(embedding),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enroll error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/verify")
async def verify(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    extract_target: bool = Form(False),
):
    """
    身份验证 (1:1)
    
    如果 extract_target=True，会先做说话人分离，
    从混合语音中提取目标用户的片段再验证。
    """
    try:
        audio_data, sr = await _load_upload(audio)
        duration = len(audio_data) / sr
        
        if len(audio_data) < sr * 0.3:
            raise HTTPException(400, "Audio too short (min 0.3s)")
        
        # 获取注册声纹
        target_emb = store.get_embedding(user_id)
        if target_emb is None:
            raise HTTPException(404, f"User '{user_id}' not enrolled")
        
        speaker_count = 1
        used_audio = audio_data
        
        if extract_target:
            # 从混合语音中提取目标说话人
            extracted, segments = engine.extract_target_speaker(
                audio_data, sr, target_emb
            )
            speaker_count = engine.get_speaker_count_estimate(audio_data, sr)
            
            if len(extracted) < sr * 0.3:
                return {
                    "success": False,
                    "score": 0.0,
                    "threshold": engine.verify_threshold,
                    "message": "Target speaker not found in audio",
                    "speaker_count": speaker_count,
                    "duration": duration,
                }
            used_audio = extracted
        
        # 提取声纹并比对
        emb = engine.extract_embedding(used_audio, sr)
        score = engine.compare(emb, target_emb)
        
        success = score >= engine.verify_threshold
        store.log_verify(user_id, success, score, duration, speaker_count)
        
        return {
            "success": success,
            "score": round(score, 4),
            "threshold": engine.verify_threshold,
            "identity": user_id if success else None,
            "speaker_count": speaker_count,
            "duration": duration,
            "extracted_duration": float(len(used_audio) / sr) if extract_target else duration,
            "message": "Verified" if success else "Rejected",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    top_k: int = Form(5),
):
    """说话人识别 (1:N)"""
    try:
        audio_data, sr = await _load_upload(audio)
        
        if len(audio_data) < sr * 0.3:
            raise HTTPException(400, "Audio too short")
        
        # 提取声纹
        emb = engine.extract_embedding(audio_data, sr)
        
        # 与所有注册用户比对
        all_embeddings = store.get_all_embeddings()
        if not all_embeddings:
            raise HTTPException(404, "No speakers enrolled")
        
        scores = {}
        for uid, ref_emb in all_embeddings:
            scores[uid] = engine.compare(emb, ref_emb)
        
        # 排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 获取名字
        speakers = {s["user_id"]: s["name"] for s in store.list_speakers()}
        
        top_results = [
            {
                "user_id": uid,
                "name": speakers.get(uid, uid),
                "score": round(sc, 4),
            }
            for uid, sc in sorted_scores[:top_k]
        ]
        
        best_uid, best_score = sorted_scores[0]
        identified = best_uid if best_score >= engine.identify_threshold else None
        
        return {
            "identity": identified,
            "name": speakers.get(best_uid, best_uid) if identified else None,
            "score": round(best_score, 4),
            "threshold": engine.identify_threshold,
            "top_k": top_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identify error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/diarize")
async def diarize(
    audio: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
):
    """说话人分离"""
    try:
        audio_data, sr = await _load_upload(audio)
        
        segments = engine.diarize(audio_data, sr, num_speakers)
        
        # 查找已注册用户
        all_embeddings = store.get_all_embeddings()
        speakers = {s["user_id"]: s["name"] for s in store.list_speakers()}
        
        seg_list = []
        for seg in segments:
            identity = None
            if all_embeddings:
                best_uid = None
                best_sim = 0
                for uid, ref_emb in all_embeddings:
                    if seg.embedding is not None:
                        sim = engine.compare(seg.embedding, ref_emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_uid = uid
                if best_sim >= engine.identify_threshold:
                    identity = best_uid
            
            seg_list.append({
                "speaker_idx": seg.speaker_idx,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.end - seg.start, 2),
                "identity": identity,
                "name": speakers.get(identity, f"Speaker_{seg.speaker_idx}") if identity else f"Speaker_{seg.speaker_idx}",
            })
        
        return {
            "total_speakers": len(set(s["speaker_idx"] for s in seg_list)),
            "total_segments": len(seg_list),
            "duration": float(len(audio_data) / sr),
            "segments": seg_list,
        }
    except Exception as e:
        logger.error(f"Diarize error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/extract_target")
async def extract_target(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
):
    """
    从混合语音中提取目标说话人的语音
    返回提取后的音频文件
    """
    try:
        audio_data, sr = await _load_upload(audio)
        target_emb = store.get_embedding(user_id)
        
        if target_emb is None:
            raise HTTPException(404, f"User '{user_id}' not enrolled")
        
        extracted, segments = engine.extract_target_speaker(audio_data, sr, target_emb)
        
        if len(extracted) == 0:
            return JSONResponse({
                "success": False,
                "message": "Target speaker not found",
            })
        
        # 返回提取的音频
        import torch
        import torchaudio
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(tmp_path, torch.from_numpy(extracted).unsqueeze(0).float(), sr)
        
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()
        import os
        os.unlink(tmp_path)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "X-Segments": str(len(segments)),
                "X-Duration": str(round(len(extracted) / sr, 2)),
                "X-Original-Duration": str(round(len(audio_data) / sr, 2)),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extract target error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/speakers")
async def list_speakers():
    """列出所有已注册用户"""
    return {"speakers": store.list_speakers()}


@router.delete("/speakers/{user_id}")
async def delete_speaker(user_id: str):
    """删除用户声纹"""
    success = store.delete_speaker(user_id)
    if not success:
        raise HTTPException(404, "User not found")
    return {"success": True, "user_id": user_id}


@router.get("/stats")
async def stats():
    """系统统计"""
    return store.get_stats()


@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": engine._classifier is not None,
        "speakers": len(store._cache),
    }
