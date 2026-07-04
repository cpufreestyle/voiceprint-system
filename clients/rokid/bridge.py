#!/usr/bin/env python3
"""
Rokid 眼镜 Python 桥接客户端

通过手机端 Rokid Bridge APK 获取眼镜音频，
转发到声纹识别服务器进行验证。

工作流程:
  Rokid眼镜 ──蓝牙/WiFi──> 手机Bridge APK ──HTTP──> 本脚本 ──HTTP──> 声纹服务器

使用方式:
  # 注册
  python bridge.py --phone 192.168.1.100:9090 enroll --user alice --name Alice

  # 验证（带说话人分离，只识别你的声音）
  python bridge.py --phone 192.168.1.100:9090 verify --user alice

  # 持续监听
  python bridge.py --phone 192.168.1.100:9090 watch --user alice

依赖: pip install httpx
"""

import argparse
import io
import sys
import time
import wave
import json
import httpx

# 默认配置
DEFAULT_SERVER = "http://localhost:8700"
PHONE_BRIDGE_PORT = 9090


class RokidBridge:
    """与手机端 Rokid Bridge APK 通信"""
    
    def __init__(self, phone_ip: str, phone_port: int = PHONE_BRIDGE_PORT):
        self.base_url = f"http://{phone_ip}:{phone_port}"
        self.client = httpx.Client(timeout=30.0)
    
    def get_status(self) -> dict:
        """获取眼镜连接状态"""
        try:
            resp = self.client.get(f"{self.base_url}/status")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    def start_recording(self, duration: float = 5.0) -> bytes:
        """开始录音，返回 WAV 字节"""
        resp = self.client.post(
            f"{self.base_url}/record",
            params={"duration": duration},
            timeout=duration + 10
        )
        resp.raise_for_status()
        return resp.content  # WAV bytes
    
    def start_vad_recording(self, silence_duration: float = 1.5, 
                            max_duration: float = 30.0) -> bytes:
        """VAD 录音（检测到静音自动停止）"""
        resp = self.client.post(
            f"{self.base_url}/record_vad",
            params={"silence": silence_duration, "max_duration": max_duration},
            timeout=max_duration + 10
        )
        resp.raise_for_status()
        return resp.content
    
    def send_ar_notification(self, text: str, color: str = "white"):
        """在眼镜 AR 显示屏上推送通知"""
        try:
            self.client.post(
                f"{self.base_url}/ar_notify",
                json={"text": text, "color": color}
            )
        except Exception:
            pass  # 非关键功能


class VoicePrintClient:
    """声纹识别服务器客户端"""
    
    def __init__(self, server_url: str):
        self.server = server_url
        self.client = httpx.Client(timeout=30.0)
    
    def enroll(self, user_id: str, name: str, audio: bytes) -> dict:
        """注册声纹"""
        resp = self.client.post(
            f"{self.server}/api/enroll",
            files={"audio": ("voice.wav", audio, "audio/wav")},
            data={"user_id": user_id, "name": name}
        )
        resp.raise_for_status()
        return resp.json()
    
    def verify(self, user_id: str, audio: bytes, extract: bool = True) -> dict:
        """验证身份"""
        resp = self.client.post(
            f"{self.server}/api/verify",
            files={"audio": ("voice.wav", audio, "audio/wav")},
            data={"user_id": user_id, "extract_target": str(extract).lower()}
        )
        resp.raise_for_status()
        return resp.json()
    
    def identify(self, audio: bytes, top_k: int = 5) -> dict:
        """识别说话人"""
        resp = self.client.post(
            f"{self.server}/api/identify",
            files={"audio": ("voice.wav", audio, "audio/wav")},
            data={"top_k": str(top_k)}
        )
        resp.raise_for_status()
        return resp.json()
    
    def list_speakers(self) -> dict:
        """列出已注册用户"""
        resp = self.client.get(f"{self.server}/api/speakers")
        resp.raise_for_status()
        return resp.json()
    
    def health(self) -> dict:
        """健康检查"""
        try:
            resp = self.client.get(f"{self.server}/api/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}


def cmd_enroll(args, bridge: RokidBridge, vp: VoicePrintClient):
    """注册声纹"""
    bridge.send_ar_notification("🎙️ 注册声纹中...", "cyan")
    print(f"🎤 Recording {args.duration}s from Rokid glasses...")
    
    audio = bridge.start_recording(args.duration)
    
    bridge.send_ar_notification("⏳ 处理中...", "yellow")
    result = vp.enroll(args.user, args.name or args.user, audio)
    
    if result.get("success"):
        bridge.send_ar_notification(f"✅ {args.user} 注册成功", "green")
        print(f"✅ Enrolled '{args.user}' ({result.get('name')})")
        print(f"   Duration: {result.get('duration')}s | Dim: {result.get('embedding_dim')}")
    else:
        bridge.send_ar_notification("❌ 注册失败", "red")
        print(f"❌ Failed: {result}")


def cmd_verify(args, bridge: RokidBridge, vp: VoicePrintClient):
    """验证身份"""
    bridge.send_ar_notification("🎙️ 请说话...", "cyan")
    print(f"🎤 Recording {args.duration}s from Rokid glasses...")
    
    audio = bridge.start_recording(args.duration)
    
    bridge.send_ar_notification("⏳ 验证中...", "yellow")
    result = vp.verify(args.user, audio, extract=True)
    
    if result.get("success"):
        bridge.send_ar_notification(f"✅ 欢迎 {result.get('identity')}", "green")
        print(f"✅ Verified: {result.get('identity')}")
        print(f"   Score: {result.get('score')} | Speakers: {result.get('speaker_count')}")
    else:
        bridge.send_ar_notification("❌ 验证失败", "red")
        print(f"❌ Rejected: {result.get('message')}")
        print(f"   Score: {result.get('score')} (threshold: {result.get('threshold')})")
        if result.get("speaker_count", 1) > 1:
            print(f"   ({result.get('speaker_count')} speakers - target extraction applied)")


def cmd_watch(args, bridge: RokidBridge, vp: VoicePrintClient):
    """持续监听"""
    bridge.send_ar_notification("👁️ 监听中...", "blue")
    print(f"👁️  Watch mode - verifying '{args.user}' via Rokid glasses")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            bridge.send_ar_notification("🎙️ 请说话...", "cyan")
            audio = bridge.start_vad_recording(silence_duration=1.5, max_duration=10.0)
            
            if len(audio) < 1000:
                continue
            
            result = vp.verify(args.user, audio, extract=True)
            
            ts = time.strftime("%H:%M:%S")
            if result.get("success"):
                bridge.send_ar_notification(f"✅ {result.get('identity')}", "green")
                print(f"  [{ts}] ✅ {result.get('identity')} (score: {result.get('score')})")
            else:
                bridge.send_ar_notification("❌ 未识别", "red")
                print(f"  [{ts}] ❌ Denied (score: {result.get('score')})")
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        bridge.send_ar_notification("👋 已停止", "purple")
        print("\n👋 Stopped")


def cmd_status(args, bridge: RokidBridge, vp: VoicePrintClient):
    """查看状态"""
    print("=== Rokid Bridge Status ===")
    status = bridge.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    print("\n=== VoicePrint Server ===")
    health = vp.health()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    print("\n=== Enrolled Speakers ===")
    speakers = vp.list_speakers()
    for s in speakers.get("speakers", []):
        print(f"  {s['user_id']} ({s['name']}) - {s['sample_count']} sample(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Rokid Glasses VoicePrint Bridge Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phone", required=True, 
                        help=f"Phone bridge IP:port (e.g. 192.168.1.100:{PHONE_BRIDGE_PORT})")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"VoicePrint server URL (default: {DEFAULT_SERVER})")
    
    sub = parser.add_subparsers(dest="command")
    
    p = sub.add_parser("enroll", help="Enroll voiceprint")
    p.add_argument("--user", required=True)
    p.add_argument("--name")
    p.add_argument("--duration", type=float, default=5.0)
    p.set_defaults(func=cmd_enroll)
    
    p = sub.add_parser("verify", help="Verify identity")
    p.add_argument("--user", required=True)
    p.add_argument("--duration", type=float, default=5.0)
    p.set_defaults(func=cmd_verify)
    
    p = sub.add_parser("watch", help="Continuous watch")
    p.add_argument("--user", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=cmd_watch)
    
    p = sub.add_parser("status", help="Show status")
    p.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    bridge = RokidBridge(args.phone)
    vp = VoicePrintClient(args.server)
    
    args.func(args, bridge, vp)


if __name__ == "__main__":
    main()
