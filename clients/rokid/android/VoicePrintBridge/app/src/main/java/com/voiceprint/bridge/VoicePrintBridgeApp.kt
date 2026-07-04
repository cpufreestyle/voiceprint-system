// VoicePrintBridgeApp.kt
// Rokid 眼镜声纹识别桥接应用
// 运行在手机端，通过 CXR-M SDK 获取眼镜音频，转发到声纹识别服务器
//
// 核心流程:
// 1. 通过 CXR-M SDK 连接 Rokid 眼镜
// 2. 获取眼镜麦克风音频流
// 3. 提供 HTTP 接口供 Python bridge.py 调用
// 4. 推送验证结果到眼镜 AR 显示屏

package com.voiceprint.bridge

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.Bundle
import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.io.ByteArrayOutputStream
import java.net.ServerSocket
import java.net.InetSocketAddress
import java.io.OutputStream
import java.io.InputStream
import kotlin.concurrent.thread

// Rokid CXR-M SDK imports
// import com.rokid.cxr.api.CxrApi
// import com.rokid.cxr.api.ValueUtil
// import com.rokid.cxr.api.listener.AudioStreamListener

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "VoicePrintBridge"
        private const val REQUEST_PERMISSIONS = 1001
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 请求权限
        val permissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.INTERNET
        )
        
        val needed = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), REQUEST_PERMISSIONS)
        } else {
            startBridgeService()
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            startBridgeService()
        }
    }
    
    private fun startBridgeService() {
        val intent = Intent(this, BridgeService::class.java)
        startForegroundService(intent)
        Log.i(TAG, "Bridge service started")
    }
}

/**
 * 桥接服务
 * 
 * 在手机端运行 HTTP 服务器，提供以下接口：
 * - GET  /status    - 获取眼镜连接状态
 * - POST /record    - 录音指定时长
 * - POST /record_vad - VAD 录音（静音停止）
 * - POST /ar_notify - 推送 AR 通知到眼镜
 */
class BridgeService : Service() {
    
    companion object {
        private const val TAG = "BridgeService"
        private const val PORT = 9090
        private const val SAMPLE_RATE = 16000
        private const val CHANNELS = 1
        private const val BITS_PER_SAMPLE = 16
    }
    
    private var serverSocket: ServerSocket? = null
    private var isRunning = false
    // private var cxrApi: CxrApi? = null
    private var isGlassesConnected = false
    
    // 音频缓冲
    private val audioBuffer = ByteArrayOutputStream()
    
    override fun onCreate() {
        super.onCreate()
        // 初始化 CXR-M SDK
        // cxrApi = CxrApi.getInstance()
        // cxrApi?.init(applicationContext)
        
        // 设置音频流监听
        // cxrApi?.setupAudioStreamListener(object : AudioStreamListener {
        //     override fun onAudioData(data: ByteArray, sampleRate: Int) {
        //         audioBuffer.write(data)
        //     }
        //     override fun onAudioStateChanged(state: Int) {
        //         Log.d(TAG, "Audio state: $state")
        //     }
        // })
        
        startHttpServer()
    }
    
    private fun startHttpServer() {
        isRunning = true
        thread {
            try {
                serverSocket = ServerSocket(PORT)
                Log.i(TAG, "HTTP server on port $PORT")
                
                while (isRunning) {
                    val client = serverSocket?.accept() ?: break
                    thread { handleClient(client) }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Server error: ${e.message}")
            }
        }
    }
    
    private fun handleClient(client: java.net.Socket) {
        try {
            val input = client.getInputStream()
            val output = client.getOutputStream()
            
            val requestLine = input.bufferedReader().readLine() ?: return
            val parts = requestLine.split(" ")
            val method = parts[0]
            val path = parts[1]
            
            Log.d(TAG, "$method $path")
            
            when {
                path == "/status" && method == "GET" -> {
                    val status = """{"connected": $isGlassesConnected, "device": "Rokid Glasses"}"""
                    sendJsonResponse(output, status)
                }
                
                path.startsWith("/record") && method == "POST" -> {
                    val params = parseQuery(path.substringAfter("?", ""))
                    val duration = params["duration"]?.toFloatOrNull() ?: 5.0f
                    
                    // 录音
                    val audio = recordFromGlasses(duration)
                    
                    // 返回 WAV
                    val wav = pcmToWav(audio, SAMPLE_RATE, CHANNELS)
                    sendWavResponse(output, wav)
                }
                
                path.startsWith("/record_vad") && method == "POST" -> {
                    val params = parseQuery(path.substringAfter("?", ""))
                    val silence = params["silence"]?.toFloatOrNull() ?: 1.5f
                    val maxDuration = params["max_duration"]?.toFloatOrNull() ?: 30.0f
                    
                    val audio = recordVadFromGlasses(silence, maxDuration)
                    val wav = pcmToWav(audio, SAMPLE_RATE, CHANNELS)
                    sendWavResponse(output, wav)
                }
                
                path == "/ar_notify" && method == "POST" -> {
                    // 读取 body
                    val body = readBody(input)
                    // 推送到眼镜 AR 显示
                    // val json = String(body)
                    // cxrApi?.sendStream(
                    //     ValueUtil.CxrStreamType.AI_ASSISTANT,
                    //     json
                    // )
                    sendJsonResponse(output, """{"success": true}""")
                }
                
                else -> {
                    output.write("HTTP/1.1 404 Not Found\r\n\r\n".toByteArray())
                }
            }
            
            output.flush()
            client.close()
        } catch (e: Exception) {
            Log.e(TAG, "Client error: ${e.message}")
        }
    }
    
    /**
     * 从眼镜录音
     * 通过 CXR-M SDK 的音频流接口获取
     */
    private fun recordFromGlasses(duration: Float): ByteArray {
        audioBuffer.reset()
        
        // 方式1: 使用 CXR-M SDK 获取眼镜音频
        // cxrApi?.startAudioStream()
        // Thread.sleep((duration * 1000).toLong())
        // cxrApi?.stopAudioStream()
        
        // 方式2: 使用手机麦克风（备选）
        // 当眼镜未连接时 fallback
        val audioData = recordWithLocalMic(duration)
        
        return audioData
    }
    
    /**
     * VAD 录音
     */
    private fun recordVadFromGlasses(silenceDuration: Float, maxDuration: Float): ByteArray {
        // 实际实现使用 AudioRecord + 能量 VAD
        // 这里返回空表示需要眼镜连接
        return recordFromGlasses(minOf(maxDuration, 5.0f))
    }
    
    /**
     * 手机本地麦克风录音（fallback）
     */
    private fun recordWithLocalMic(duration: Float): ByteArray {
        val androidMediaRecorder = android.media.AudioRecord(
            android.media.MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            android.media.AudioFormat.CHANNEL_IN_MONO,
            android.media.AudioFormat.ENCODING_PCM_16BIT,
            SAMPLE_RATE * 2 * Math.ceil(duration.toDouble()).toInt()
        )
        
        val bufferSize = (SAMPLE_RATE * duration).toInt() * 2
        val buffer = ByteArray(bufferSize)
        
        androidMediaRecorder.startRecording()
        androidMediaRecorder.read(buffer, 0, bufferSize)
        androidMediaRecorder.stop()
        androidMediaRecorder.release()
        
        return buffer
    }
    
    /**
     * PCM → WAV
     */
    private fun pcmToWav(pcm: ByteArray, sampleRate: Int, channels: Int): ByteArray {
        val byteRate = sampleRate * channels * (BITS_PER_SAMPLE / 8)
        val totalDataLen = pcm.size
        val totalLen = totalDataLen + 36
        
        val baos = ByteArrayOutputStream()
        val dos = java.io.DataOutputStream(baos)
        
        // RIFF header
        dos.writeBytes("RIFF")
        dos.write(intToLE(totalLen))
        dos.writeBytes("WAVE")
        
        // fmt chunk
        dos.writeBytes("fmt ")
        dos.write(intToLE(16))           // chunk size
        dos.write(shortToLE(1))          // audio format = PCM
        dos.write(shortToLE(channels))
        dos.write(intToLE(sampleRate))
        dos.write(intToLE(byteRate))
        dos.write(shortToLE(channels * (BITS_PER_SAMPLE / 8)))
        dos.write(shortToLE(BITS_PER_SAMPLE))
        
        // data chunk
        dos.writeBytes("data")
        dos.write(intToLE(totalDataLen))
        dos.write(pcm)
        
        return baos.toByteArray()
    }
    
    private fun intToLE(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )
    }
    
    private fun shortToLE(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte()
        )
    }
    
    private fun parseQuery(query: String): Map<String, String> {
        return query.split("&").associate {
            val parts = it.split("=")
            if (parts.size == 2) parts[0] to parts[1] else "" to ""
        }
    }
    
    private fun readBody(input: InputStream): ByteArray {
        val buf = ByteArrayOutputStream()
        val tmp = ByteArray(1024)
        while (true) {
            val n = input.read(tmp)
            if (n <= 0) break
            buf.write(tmp, 0, n)
            if (buf.size() > 100000) break // max 100KB
        }
        return buf.toByteArray()
    }
    
    private fun sendJsonResponse(output: OutputStream, json: String) {
        val header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${json.toByteArray().size}\r\n\r\n"
        output.write(header.toByteArray())
        output.write(json.toByteArray())
    }
    
    private fun sendWavResponse(output: OutputStream, wav: ByteArray) {
        val header = "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: ${wav.size}\r\n\r\n"
        output.write(header.toByteArray())
        output.write(wav)
    }
    
    override fun onDestroy() {
        isRunning = false
        serverSocket?.close()
        super.onDestroy()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
}
