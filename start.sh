#!/bin/bash
# VoicePrint System - 快速启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔊 VoicePrint Recognition System"
echo "================================"

# 检查依赖
echo "1. Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null || true

# 安装 sounddevice（客户端录音用）
pip install -q sounddevice 2>/dev/null || true

echo "2. Starting server on http://localhost:8700"
echo "   API docs: http://localhost:8700/docs"
echo "   Press Ctrl+C to stop"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8700 --reload
