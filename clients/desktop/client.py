#!/usr/bin/env python3
"""
桌面机器人客户端 - 与 AI 眼镜客户端共用核心逻辑
增加：语音活动检测启动、LED 状态反馈、唤醒词检测

使用方式:
  python client.py enroll --user boss --name "Boss"
  python client.py verify --user boss
  python client.py watch --user boss --with-led
"""

import sys
import os
import time
import threading
from pathlib import Path

# 共用眼镜客户端的核心功能
sys.path.insert(0, str(Path(__file__).parent.parent / "glasses"))
from client import record_audio, record_until_silence, send_request, DEFAULT_SERVER

import argparse
import numpy as np


class LEDIndicator:
    """LED 状态指示器（桌面机器人用）"""
    
    # 颜色状态
    IDLE = ("idle", "blue", 0.3)       # 待命：蓝色慢闪
    LISTENING = ("listening", "cyan", 1.0)  # 监听：青色常亮
    PROCESSING = ("processing", "yellow", 2.0)  # 处理：黄色快闪
    SUCCESS = ("success", "green", 1.0)  # 成功：绿色
    FAILED = ("failed", "red", 1.0)     # 失败：红色
    NO_SPEECH = ("no_speech", "purple", 0.5)  # 无语音：紫色
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._thread = None
        self._stop = False
        self._current = self.IDLE
    
    def set_state(self, state):
        """设置 LED 状态"""
        self._current = state
        if self.enabled:
            name, color, speed = state
            # 实际硬件对接：发送 GPIO/串口指令
            # 这里用终端颜色模拟
            colors = {
                "blue": "\033[94m", "cyan": "\033[96m",
                "yellow": "\033[93m", "green": "\033[92m",
                "red": "\033[91m", "purple": "\033[95m",
            }
            reset = "\033[0m"
            print(f"  💡 LED: {colors.get(color, '')}● {name}{reset}")
    
    def blink_loop(self):
        """LED 闪烁循环"""
        while not self._stop:
            name, color, speed = self._current
            time.sleep(1.0 / speed if speed > 0 else 1.0)
    
    def start(self):
        if self.enabled:
            self._thread = threading.Thread(target=self.blink_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        self._stop = True


def cmd_enroll(args):
    """注册声纹"""
    led = LEDIndicator(args.with_led)
    led.start()
    
    led.set_state(LEDIndicator.LISTENING)
    audio = record_audio(args.duration)
    led.set_state(LEDIndicator.PROCESSING)
    
    result = send_request("enroll", audio, args.server,
                          user_id=args.user, name=args.name or args.user)
    
    if result and result.get("success"):
        led.set_state(LEDIndicator.SUCCESS)
        print(f"✅ Enrolled '{args.user}' ({result.get('name')})")
    else:
        led.set_state(LEDIndicator.FAILED)
        print(f"❌ Enroll failed")
    
    time.sleep(1)
    led.stop()


def cmd_verify(args):
    """验证身份"""
    led = LEDIndicator(args.with_led)
    led.start()
    
    led.set_state(LEDIndicator.LISTENING)
    audio = record_audio(args.duration)
    led.set_state(LEDIndicator.PROCESSING)
    
    result = send_request("verify", audio, args.server,
                          user_id=args.user, extract_target=True)
    
    if result:
        if result.get("success"):
            led.set_state(LEDIndicator.SUCCESS)
            print(f"✅ Welcome back, {result.get('identity')}!")
            print(f"   Score: {result.get('score')} | Speakers: {result.get('speaker_count')}")
        else:
            led.set_state(LEDIndicator.FAILED)
            print(f"❌ Access denied. Score: {result.get('score')}")
            if result.get("speaker_count", 1) > 1:
                print(f"   (Multiple speakers detected - target extraction applied)")
    else:
        led.set_state(LEDIndicator.FAILED)
    
    time.sleep(1)
    led.stop()


def cmd_watch(args):
    """持续监听"""
    led = LEDIndicator(args.with_led)
    led.start()
    
    print(f"🤖 Desktop robot watching for '{args.user}'")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            led.set_state(LEDIndicator.LISTENING)
            audio = record_until_silence(
                silence_duration=1.5,
                max_duration=10.0,
            )
            
            if not audio:
                led.set_state(LEDIndicator.NO_SPEECH)
                time.sleep(0.5)
                continue
            
            led.set_state(LEDIndicator.PROCESSING)
            result = send_request("verify", audio, args.server,
                                  user_id=args.user, extract_target=True)
            
            if result:
                ts = time.strftime("%H:%M:%S")
                if result.get("success"):
                    led.set_state(LEDIndicator.SUCCESS)
                    print(f"  [{ts}] ✅ {result.get('identity')} (score: {result.get('score')})")
                else:
                    led.set_state(LEDIndicator.FAILED)
                    print(f"  [{ts}] ❌ Denied (score: {result.get('score')})")
            
            time.sleep(args.interval)
            led.set_state(LEDIndicator.IDLE)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped")
        led.stop()


def main():
    parser = argparse.ArgumentParser(
        description="VoicePrint Client for Desktop Robot",
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--with-led", action="store_true", default=True,
                        help="Enable LED indicator simulation")
    
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
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
