#!/usr/bin/env python3
"""
AI 眼镜客户端 - 轻量级录音 + 声纹交互

使用方式:
  # 录音注册
  python client.py enroll --user alice --name "Alice"
  
  # 录音验证（带说话人分离，只验证你的声音）
  python client.py verify --user alice --extract
  
  # 录音识别（谁在说话）
  python client.py identify
  
  # 录音分离（多人对话分析）
  python client.py diarize
  
  # 持续监听模式（实时验证）
  python client.py watch --user alice

依赖: pip install sounddevice httpx
"""

import argparse
import io
import sys
import time
import wave
import json
import tempfile
import os
from pathlib import Path

try:
    import sounddevice as sd
    import numpy as np
    import httpx
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install sounddevice httpx numpy")
    sys.exit(1)


# ========== 配置 ==========
DEFAULT_SERVER = os.getenv("VP_SERVER", "http://localhost:8700")
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # 默认录音时长


def record_audio(duration: float = RECORD_SECONDS, sr: int = SAMPLE_RATE) -> bytes:
    """录音并返回 WAV 字节"""
    print(f"🎤 Recording {duration}s...")
    
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    
    print("✅ Recording complete")
    
    # 转 WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # float32 -> int16
        audio_int16 = (recording * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


def record_until_silence(sr: int = SAMPLE_RATE, 
                          silence_duration: float = 1.5,
                          max_duration: float = 30.0,
                          energy_threshold: float = 0.01) -> bytes:
    """录音直到检测到静音"""
    print("🎤 Listening... (speak now)")
    
    block_size = int(0.03 * sr)  # 30ms blocks
    silence_blocks = int(silence_duration / 0.03)
    max_blocks = int(max_duration / 0.03)
    
    audio_chunks = []
    silence_count = 0
    has_speech = False
    total_blocks = 0
    
    with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=block_size) as stream:
        while total_blocks < max_blocks:
            data, overflowed = stream.read(block_size)
            audio_chunks.append(data)
            total_blocks += 1
            
            energy = np.mean(data ** 2)
            
            if energy > energy_threshold:
                has_speech = True
                silence_count = 0
            elif has_speech:
                silence_count += 1
                if silence_count >= silence_blocks:
                    break
    
    if not has_speech:
        print("⚠️  No speech detected")
        return b""
    
    print(f"✅ Recorded {total_blocks * 0.03:.1f}s")
    
    recording = np.concatenate(audio_chunks).squeeze()
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        audio_int16 = (recording * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


def send_request(endpoint: str, audio_bytes: bytes, 
                 server: str = DEFAULT_SERVER, **form_fields):
    """发送请求到服务器"""
    url = f"{server}/api/{endpoint}"
    files = {"audio": ("voice.wav", audio_bytes, "audio/wav")}
    data = {k: str(v) for k, v in form_fields.items() if v is not None}
    
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, files=files, data=data)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP {e.response.status_code}: {e.response.text}")
        return None
    except httpx.ConnectError:
        print(f"❌ Cannot connect to {server}")
        print(f"   Is the server running? Start it with: python -m app.main")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ========== 命令 ==========

def cmd_enroll(args):
    """注册声纹"""
    audio = record_audio(args.duration)
    result = send_request("enroll", audio, args.server,
                          user_id=args.user, name=args.name or args.user)
    if result:
        if result.get("success"):
            print(f"✅ Enrolled '{args.user}' ({result.get('name')})")
            print(f"   Duration: {result.get('duration')}s")
            print(f"   Embedding dim: {result.get('embedding_dim')}")
        else:
            print(f"❌ Enroll failed: {result}")


def cmd_verify(args):
    """验证身份"""
    audio = record_audio(args.duration)
    result = send_request("verify", audio, args.server,
                          user_id=args.user, extract_target=args.extract)
    if result:
        if result.get("success"):
            print(f"✅ Verified: {result.get('identity')}")
            print(f"   Score: {result.get('score')} (threshold: {result.get('threshold')})")
        else:
            print(f"❌ Rejected: {result.get('message', 'score too low')}")
            print(f"   Score: {result.get('score')} (threshold: {result.get('threshold')})")
        
        if args.extract:
            print(f"   Speakers detected: {result.get('speaker_count', '?')}")
            print(f"   Extracted: {result.get('extracted_duration', 0)}s from {result.get('duration', 0)}s")


def cmd_identify(args):
    """识别说话人"""
    audio = record_audio(args.duration)
    result = send_request("identify", audio, args.server, top_k=args.top_k)
    if result:
        if result.get("identity"):
            print(f"✅ Identified: {result.get('name')} ({result.get('identity')})")
            print(f"   Score: {result.get('score')} (threshold: {result.get('threshold')})")
        else:
            print(f"❌ No match (best score: {result.get('score')} < {result.get('threshold')})")
        
        print("\nTop matches:")
        for i, match in enumerate(result.get("top_k", [])):
            marker = " ← best" if i == 0 else ""
            print(f"  {i+1}. {match['name']} ({match['user_id']}): {match['score']}{marker}")


def cmd_diarize(args):
    """说话人分离"""
    audio = record_audio(args.duration)
    result = send_request("diarize", audio, args.server, 
                          num_speakers=args.num_speakers)
    if result:
        print(f"📊 Found {result.get('total_speakers')} speakers in {result.get('duration')}s")
        print(f"   {result.get('total_segments')} segments:\n")
        
        for seg in result.get("segments", []):
            print(f"  [{seg['start']:.1f}-{seg['end']:.1f}] {seg['name']} "
                  f"({seg['duration']:.1f}s)")


def cmd_watch(args):
    """持续监听模式 - 实时验证"""
    print(f"👁️  Watch mode - continuously verifying '{args.user}'")
    print("   Press Ctrl+C to stop\n")
    
    check_interval = args.interval  # 秒
    
    try:
        while True:
            audio = record_until_silence(
                silence_duration=1.5,
                max_duration=10.0,
            )
            
            if not audio:
                continue
            
            result = send_request("verify", audio, args.server,
                                  user_id=args.user, extract_target=True)
            
            if result:
                ts = time.strftime("%H:%M:%S")
                if result.get("success"):
                    print(f"  [{ts}] ✅ {result.get('identity')} (score: {result.get('score')})")
                else:
                    print(f"  [{ts}] ❌ Not verified (score: {result.get('score')})")
                
                if result.get("speaker_count", 1) > 1:
                    print(f"         ({result.get('speaker_count')} speakers in audio)")
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped")


def cmd_list(args):
    """列出已注册用户"""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{args.server}/api/speakers")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"❌ {e}")
        return
    
    speakers = data.get("speakers", [])
    if not speakers:
        print("No speakers enrolled")
        return
    
    print(f"📋 {len(speakers)} speaker(s) enrolled:")
    for s in speakers:
        status = "🟢" if s["active"] else "🔴"
        print(f"  {status} {s['user_id']} ({s['name']}) - {s['sample_count']} sample(s), updated {s['updated_at'][:19]}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="VoicePrint Client for AI Glasses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Server URL (default: {DEFAULT_SERVER})")
    
    sub = parser.add_subparsers(dest="command", help="Command")
    
    # enroll
    p = sub.add_parser("enroll", help="Enroll voiceprint")
    p.add_argument("--user", required=True, help="User ID")
    p.add_argument("--name", help="Display name")
    p.add_argument("--duration", type=float, default=5.0, help="Recording duration (seconds)")
    p.set_defaults(func=cmd_enroll)
    
    # verify
    p = sub.add_parser("verify", help="Verify identity (1:1)")
    p.add_argument("--user", required=True, help="User ID to verify against")
    p.add_argument("--duration", type=float, default=5.0, help="Recording duration")
    p.add_argument("--extract", action="store_true", default=True,
                   help="Extract target speaker from mixed audio (enabled by default)")
    p.set_defaults(func=cmd_verify)
    
    # identify
    p = sub.add_parser("identify", help="Identify speaker (1:N)")
    p.add_argument("--duration", type=float, default=5.0, help="Recording duration")
    p.add_argument("--top-k", type=int, default=5, help="Top K results")
    p.set_defaults(func=cmd_identify)
    
    # diarize
    p = sub.add_parser("diarize", help="Speaker diarization")
    p.add_argument("--duration", type=float, default=10.0, help="Recording duration")
    p.add_argument("--num-speakers", type=int, default=None, help="Expected speaker count")
    p.set_defaults(func=cmd_diarize)
    
    # watch
    p = sub.add_parser("watch", help="Continuous watch mode")
    p.add_argument("--user", required=True, help="User ID to watch for")
    p.add_argument("--interval", type=float, default=1.0, help="Check interval (seconds)")
    p.set_defaults(func=cmd_watch)
    
    # list
    p = sub.add_parser("list", help="List enrolled speakers")
    p.set_defaults(func=cmd_list)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
