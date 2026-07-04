# Rokid 眼镜适配层

## 概述

本目录包含 Rokid AR 眼镜的声纹识别客户端适配代码。

Rokid 眼镜通过 **CXR-M SDK**（手机端）或 **CXR-S SDK**（眼镜端）开发，
本适配层提供两种接入方式：

### 1. Android 原生 APK（推荐）
- 在手机端运行，通过 CXR-M SDK 获取眼镜音频
- 通过 Wi-Fi 将音频发到声纹识别服务器
- 眼镜端显示验证结果（通过 CXR-M 推送 AR 场景）

### 2. Python 桥接（快速验证）
- 手机端运行一个轻量 HTTP 转发服务
- Python 客户端从手机端拉取音频再发给声纹服务器
- 适合开发调试

## 文件结构

```
rokid/
├── README.md           # 本文件
├── android/            # Android 原生项目
│   ├── VoicePrintApp/  # Kotlin 应用源码
│   └── build.gradle    
├── bridge.py           # Python 桥接客户端
└── rokid_audio.py      # Rokid 音频流封装
```

## 快速开始

### 方式一：Python 桥接（开发调试）

1. 手机安装 Rokid VoicePrint Bridge APK（从 android/ 目录构建）
2. 手机连上 Rokid 眼镜
3. 运行：
```bash
python bridge.py --phone-ip <手机IP> --action enroll --user alice
```

### 方式二：Android 原生（生产环境）

直接在 Android Studio 中打开 `android/VoicePrintApp/`，
构建安装到手机即可。应用会自动：
- 连接 Rokid 眼镜
- 获取眼镜麦克风音频
- 发送到声纹识别服务器
- 在眼镜 AR 显示屏上展示验证结果

## Rokid CXR-M SDK 关键接口

```kotlin
// 获取眼镜端音频流
CxrApi.getInstance().setupAudioStreamListener(object : AudioStreamListener {
    override fun onAudioData(data: ByteArray, sampleRate: Int) {
        // data 是 PCM 音频数据
        // 发送到声纹识别服务器
    }
})

// 推送 AR 显示内容到眼镜
CxrApi.getInstance().sendStream(
    ValueUtil.CxrStreamType.AI_ASSISTANT,
    jsonContent
)
```

## 权限要求

- `RECORD_AUDIO` - 麦克风录音
- `BLUETOOTH` / `BLUETOOTH_CONNECT` - 蓝牙连接眼镜
- `ACCESS_FINE_LOCATION` - 蓝牙扫描需要
- `INTERNET` - 网络通信
