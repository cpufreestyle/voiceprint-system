#!/usr/bin/env python3
"""
测试脚本 - 验证系统各组件是否正常工作
"""

import sys
import os
import time
import tempfile
import wave
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试依赖导入"""
    print("1. Testing imports...")
    
    import torch
    print(f"   ✅ torch {torch.__version__}")
    
    import speechbrain
    print(f"   ✅ speechbrain {speechbrain.__version__}")
    
    import fastapi
    print(f"   ✅ fastapi {fastapi.__version__}")
    
    import soundfile
    print(f"   ✅ soundfile")
    
    print()


def test_engine():
    """测试引擎初始化和声纹提取"""
    print("2. Testing VoicePrintEngine...")
    from app.core.engine import VoicePrintEngine
    
    engine = VoicePrintEngine(model_cache_dir="./models")
    
    # 生成测试音频（1秒正弦波模拟语音）
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 模拟语音：多频率正弦波
    audio = 0.3 * (np.sin(2 * np.pi * 200 * t) + 
                   0.5 * np.sin(2 * np.pi * 400 * t) +
                   0.3 * np.sin(2 * np.pi * 800 * t))
    audio = audio.astype(np.float32)
    
    print(f"   Loading model (first run downloads ~80MB)...")
    print(f"   This may take a minute...")
    
    # 测试声纹提取
    embedding = engine.extract_embedding(audio, sr)
    print(f"   ✅ Embedding extracted: shape={embedding.shape}")
    assert embedding.shape[0] == 192, f"Expected 192-dim, got {embedding.shape[0]}"
    
    # 测试比对
    emb2 = engine.extract_embedding(audio, sr)
    score = engine.compare(embedding, emb2)
    print(f"   ✅ Self-similarity: {score:.4f} (should be ~1.0)")
    assert score > 0.99, "Self-similarity should be near 1.0"
    
    # 测试不同音频
    audio2 = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    emb3 = engine.extract_embedding(audio2, sr)
    score2 = engine.compare(embedding, emb3)
    print(f"   ✅ Different audio similarity: {score2:.4f}")
    
    # 测试 VAD
    segments = engine._energy_vad(audio, sr)
    print(f"   ✅ VAD found {len(segments)} segments")
    
    # 测试音频统计
    stats = engine.audio_stats(audio, sr)
    print(f"   ✅ Audio stats: {stats}")
    
    print()
    return engine


def test_store():
    """测试声纹库"""
    print("3. Testing VoicePrintStore...")
    from app.core.store import VoicePrintStore
    from app.core.engine import VoicePrintEngine
    
    store = VoicePrintStore(db_path="./data/test_voiceprint.db")
    
    # 生成假声纹
    emb = np.random.randn(192).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    
    # 注册
    success = store.enroll("test_user", "Test User", emb)
    assert success, "Enroll failed"
    print("   ✅ Enrolled test_user")
    
    # 查询
    retrieved = store.get_embedding("test_user")
    assert retrieved is not None, "Get embedding failed"
    print(f"   ✅ Retrieved embedding: shape={retrieved.shape}")
    
    # 列表
    speakers = store.list_speakers()
    assert len(speakers) >= 1
    print(f"   ✅ Listed {len(speakers)} speaker(s)")
    
    # 更新（多次注册）
    emb2 = np.random.randn(192).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    store.enroll("test_user", "Test User", emb2)
    speakers = store.list_speakers()
    for s in speakers:
        if s["user_id"] == "test_user":
            assert s["sample_count"] == 2, f"Expected 2 samples, got {s['sample_count']}"
            print(f"   ✅ Updated: sample_count={s['sample_count']}")
    
    # 删除
    store.delete_speaker("test_user")
    assert store.get_embedding("test_user") is None
    print("   ✅ Deleted test_user")
    
    # 清理测试数据库
    os.unlink("./data/test_voiceprint.db")
    
    print()
    return store


def test_diarization():
    """测试说话人分离"""
    print("4. Testing diarization...")
    from app.core.engine import VoicePrintEngine
    
    engine = VoicePrintEngine(model_cache_dir="./models")
    sr = 16000
    
    # 模拟两人对话：不同频率交替
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    
    # 说话人A：0-3秒，7-8秒
    mask_a = ((t >= 0) & (t < 3)) | ((t >= 7) & (t < 8))
    audio[mask_a] = 0.3 * (np.sin(2 * np.pi * 200 * t[mask_a]) + 
                            0.5 * np.sin(2 * np.pi * 400 * t[mask_a]))
    
    # 说话人B：3.5-6.5秒
    mask_b = (t >= 3.5) & (t < 6.5)
    audio[mask_b] = 0.3 * (np.sin(2 * np.pi * 300 * t[mask_b]) + 
                            0.5 * np.sin(2 * np.pi * 600 * t[mask_b]))
    
    print(f"   Running diarization on {duration}s simulated 2-speaker audio...")
    segments = engine.diarize(audio, sr)
    
    speaker_count = len(set(s.speaker_idx for s in segments))
    print(f"   ✅ Found {speaker_count} speaker(s), {len(segments)} segment(s)")
    
    for seg in segments:
        print(f"      Speaker {seg.speaker_idx}: [{seg.start:.1f}-{seg.end:.1f}] ({seg.end-seg.start:.1f}s)")
    
    print()


def main():
    print("=" * 50)
    print("VoicePrint System - Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_imports()
        test_engine()
        test_store()
        test_diarization()
        
        print("=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("  1. Start server: ./start.sh")
        print("  2. Enroll: python clients/glasses/client.py enroll --user alice --name Alice")
        print("  3. Verify: python clients/glasses/client.py verify --user alice")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
