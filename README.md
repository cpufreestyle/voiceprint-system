# VoicePrint System

声纹识别系统 — 基于 SpeechBrain ECAPA-TDNN，支持注册/验证/识别/说话人分离。

## 功能

- **注册** (`/enroll`): 上传音频 + 用户信息，提取 192 维声纹向量入库
- **验证** (`/verify`): 1:1 说话人确认
- **识别** (`/identify`): 1:N 说话人识别
- **分离** (`/diarize`): 多人音频中分离说话人
- **目标提取** (`/extract_target`): 从混合音频中提取目标说话人声纹

## 技术栈

- **模型**: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)
- **框架**: FastAPI + uvicorn
- **音频处理**: librosa + soundfile
- **分离**: pyannote.audio

## 快速开始

```bash
pip install -r requirements.txt
python -m app.main
# 服务启动在 http://localhost:8000
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/enroll | 注册声纹 |
| POST | /api/verify | 验证身份 (1:1) |
| POST | /api/identify | 识别说话人 (1:N) |
| POST | /api/diarize | 说话人分离 |
| POST | /api/extract_target | 提取目标说话人 |
| GET  | /api/speakers | 列出已注册说话人 |
| GET  | /api/stats | 系统统计 |
| GET  | /health | 健康检查 |

## 客户端

- **AI 眼镜 (Rokid)**: `clients/rokid/bridge.py` — Python 桥接模块
- **桌面机器人**: `clients/desktop/client.py` — Python 客户端
- **Android (VisionLink)**: 端侧 ONNX 推理，见 [visionlink-android](https://github.com/cpufreestyle/visionlink-android)

## License

MIT
