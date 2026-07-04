"""
VoicePrint Engine - 核心引擎
基于 ECAPA-TDNN 的声纹特征提取 + 说话人分离 + 身份验证
"""

import os
import io
import wave
import tempfile
import numpy as np
import torchaudio
import torch
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """说话人片段"""
    speaker_idx: int          # 分离出的说话人索引
    start: float              # 开始时间(秒)
    end: float                # 结束时间(秒)
    audio: np.ndarray         # 该片段的音频数据
    embedding: Optional[np.ndarray] = None  # 声纹向量
    identity: Optional[str] = None          # 识别出的身份


@dataclass
class VerifyResult:
    """验证结果"""
    success: bool
    score: float
    threshold: float
    identity: Optional[str] = None
    message: str = ""


@dataclass
class IdentifyResult:
    """识别结果"""
    identity: Optional[str]
    score: float
    threshold: float
    all_scores: Dict[str, float] = field(default_factory=dict)


class VoicePrintEngine:
    """
    声纹识别引擎
    
    功能:
    1. 声纹注册 - 提取声纹向量并存库
    2. 身份验证 - 1:1 比对，确认是否是声称的用户
    3. 说话人识别 - 1:N 辨认
    4. 说话人分离 - 从多人对话中分离各说话人
    5. 目标说话人提取 - 从混合语音中只提取指定用户的语音
    """
    
    # SpeechBrain 预训练模型
    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._classifier = None
        self._embedding_dim = 192  # ECAPA-TDNN 输出维度
        
        # 验证阈值（余弦相似度，越高越相似）
        self.verify_threshold = 0.50    # 1:1 验证阈值
        self.identify_threshold = 0.45  # 1:N 识别阈值
        
        # 分离参数
        self.min_segment_duration = 1.0   # 最短片段(秒)
        self.vad_threshold = 0.02         # 语音活动检测阈值
        
    @property
    def classifier(self):
        """懒加载 SpeechBrain 模型"""
        if self._classifier is None:
            logger.info("Loading ECAPA-TDNN model (first run downloads ~80MB)...")
            from speechbrain.inference.speaker import SpeakerRecognition
            self._classifier = SpeakerRecognition.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=str(self.model_cache_dir / "ecapa-voxceleb"),
                run_opts={"device": "cpu"}
            )
            logger.info("Model loaded successfully.")
        return self._classifier
    
    # ========== 音频处理 ==========
    
    def _load_audio(self, audio_path: str or io.BytesIO, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """加载音频文件，返回 mono numpy 数组"""
        if isinstance(audio_path, io.BytesIO):
            # 从内存加载
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_path.getvalue())
                tmp_path = tmp.name
            try:
                wav, sr = torchaudio.load(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            wav, sr = torchaudio.load(audio_path)
        
        # 转单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # 重采样到 16kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr
        
        return wav.squeeze(0).numpy(), sr
    
    def _save_wav(self, audio: np.ndarray, sr: int, path: str):
        """保存 wav 文件"""
        torchaudio.save(path, torch.from_numpy(audio).unsqueeze(0), sr)
    
    def _to_wav_bytes(self, audio: np.ndarray, sr: int) -> io.BytesIO:
        """转成 wav BytesIO"""
        buf = io.BytesIO()
        self._save_wav(audio, sr, buf.name if hasattr(buf, 'name') else "/tmp/_tmp.wav")
        # 用 torchaudio 保存到临时文件再读
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, torch.from_numpy(audio).unsqueeze(0).float(), sr)
            tmp.seek(0)
            buf = io.BytesIO(tmp.name)
        # 简化：直接用临时文件
        tmp_path = "/tmp/_vp_segment.wav"
        torchaudio.save(tmp_path, torch.from_numpy(audio).unsqueeze(0).float(), sr)
        with open(tmp_path, 'rb') as f:
            return io.BytesIO(f.read())
    
    # ========== 声纹提取 ==========
    
    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """提取声纹向量 (192维)"""
        # SpeechBrain 期望 [batch, time] 的 tensor
        wav_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(wav_tensor)
        return embedding.squeeze().cpu().numpy()
    
    def extract_embedding_from_file(self, audio_path: str) -> np.ndarray:
        """从音频文件提取声纹向量"""
        audio, sr = self._load_audio(audio_path)
        return self.extract_embedding(audio, sr)
    
    # ========== 声纹比对 ==========
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """余弦相似度 [-1, 1]"""
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm == 0:
            return 0.0
        return float(dot / norm)
    
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """比对两个声纹向量，返回相似度"""
        return self.cosine_similarity(emb1, emb2)
    
    # ========== 说话人分离 ==========
    
    def _energy_vad(self, audio: np.ndarray, sr: int, 
                    frame_len: float = 0.03, hop: float = 0.01) -> List[Tuple[float, float]]:
        """
        基于能量的语音活动检测 (VAD)
        返回语音片段列表 [(start, end), ...]
        """
        frame_size = int(frame_len * sr)
        hop_size = int(hop * sr)
        
        frames = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.mean(frame ** 2)
            frames.append((i / sr, (i + frame_size) / sr, energy))
        
        if not frames:
            return []
        
        # 动态阈值
        energies = np.array([f[2] for f in frames])
        threshold = max(self.vad_threshold, np.percentile(energies, 30))
        
        # 找语音段
        segments = []
        in_speech = False
        seg_start = 0
        
        for start, end, energy in frames:
            if energy > threshold:
                if not in_speech:
                    in_speech = True
                    seg_start = start
            else:
                if in_speech:
                    in_speech = False
                    if end - seg_start >= self.min_segment_duration:
                        segments.append((seg_start, end))
        
        if in_speech:
            last_end = frames[-1][1]
            if last_end - seg_start >= self.min_segment_duration:
                segments.append((seg_start, last_end))
        
        return segments
    
    def _split_segments(self, audio: np.ndarray, sr: int, 
                        max_segment_len: float = 10.0) -> List[Tuple[float, float, np.ndarray]]:
        """
        将长音频切分成适合处理的片段
        返回 [(start, end, audio_chunk), ...]
        """
        # 先做 VAD 分段
        vad_segments = self._energy_vad(audio, sr)
        
        if not vad_segments:
            # VAD 没检测到语音，整段处理
            vad_segments = [(0, len(audio) / sr)]
        
        # 进一步切分过长的片段
        result = []
        for start, end in vad_segments:
            seg_audio = audio[int(start * sr):int(end * sr)]
            duration = end - start
            
            if duration <= max_segment_len:
                result.append((start, end, seg_audio))
            else:
                # 滑窗切分
                chunk_size = int(max_segment_len * sr)
                hop_size = int(max_segment_len * 0.8 * sr)  # 80% overlap
                for i in range(0, len(seg_audio) - chunk_size + 1, hop_size):
                    chunk = seg_audio[i:i + chunk_size]
                    chunk_start = start + i / sr
                    chunk_end = chunk_start + len(chunk) / sr
                    result.append((chunk_start, chunk_end, chunk))
        
        return result
    
    def diarize(self, audio: np.ndarray, sr: int = 16000, 
                num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        """
        说话人分离：从音频中分离出不同说话人
        
        使用策略：
        1. VAD 切分语音段
        2. 对每段提取声纹向量
        3. 聚类分组（不需要预知说话人数量）
        
        Args:
            audio: 音频数据
            sr: 采样率
            num_speakers: 预期说话人数量（None则自动估计）
        
        Returns:
            SpeakerSegment 列表
        """
        logger.info(f"Diarizing audio: {len(audio)/sr:.1f}s")
        
        # 1. 切分片段
        segments = self._split_segments(audio, sr, max_segment_len=8.0)
        logger.info(f"VAD split into {len(segments)} segments")
        
        if len(segments) == 0:
            return []
        
        # 2. 提取每段声纹
        embeddings = []
        for start, end, seg_audio in segments:
            if len(seg_audio) < int(0.5 * sr):  # 太短跳过
                continue
            emb = self.extract_embedding(seg_audio, sr)
            embeddings.append(emb)
        
        if len(embeddings) == 0:
            return []
        
        embeddings_array = np.array(embeddings)
        
        # 3. 聚类
        speaker_labels = self._cluster_embeddings(embeddings_array, num_speakers)
        
        # 4. 合并同一说话人的相邻片段
        results = []
        for i, (start, end, seg_audio) in enumerate(segments):
            if i >= len(speaker_labels):
                break
            label = speaker_labels[i]
            emb = embeddings[i]
            
            # 尝试与上一个片段合并
            if results and results[-1].speaker_idx == label:
                # 合并音频
                prev = results[-1]
                merged_audio = np.concatenate([prev.audio, seg_audio])
                merged_emb = self.extract_embedding(merged_audio, sr)
                prev.end = end
                prev.audio = merged_audio
                prev.embedding = merged_emb
            else:
                results.append(SpeakerSegment(
                    speaker_idx=label,
                    start=start,
                    end=end,
                    audio=seg_audio,
                    embedding=emb
                ))
        
        logger.info(f"Diarization found {len(set(speaker_labels))} speakers, {len(results)} segments")
        return results
    
    def _cluster_embeddings(self, embeddings: np.ndarray, 
                            num_speakers: Optional[int] = None) -> List[int]:
        """
        聚类声纹向量
        
        使用层次聚类，如果未指定说话人数量则自动估计
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        
        if len(embeddings) <= 1:
            return [0] * len(embeddings)
        
        # 计算余弦距离矩阵
        dist_matrix = pdist(embeddings, metric='cosine')
        
        # 层次聚类
        Z = linkage(dist_matrix, method='average')
        
        if num_speakers is None:
            # 自动估计说话人数量：基于距离阈值
            # 余弦距离 < 0.5 认为是同一说话人
            labels = fcluster(Z, t=0.5, criterion='distance')
        else:
            labels = fcluster(Z, t=num_speakers, criterion='maxclust')
        
        # 转为 0-indexed
        labels = [l - 1 for l in labels]
        return labels
    
    # ========== 目标说话人提取 ==========
    
    def extract_target_speaker(self, audio: np.ndarray, sr: int,
                                target_embedding: np.ndarray,
                                min_similarity: float = 0.35) -> Tuple[np.ndarray, List[SpeakerSegment]]:
        """
        从混合语音中提取目标说话人的语音
        
        策略：
        1. 说话人分离
        2. 对每个说话人片段比对目标声纹
        3. 合并属于目标说话人的片段
        
        Args:
            audio: 混合音频
            sr: 采样率
            target_embedding: 目标用户的声纹向量
            min_similarity: 最低相似度阈值
        
        Returns:
            (提取的音频, 匹配的片段列表)
        """
        # 1. 分离说话人
        segments = self.diarize(audio, sr)
        
        # 2. 匹配目标说话人
        target_segments = []
        for seg in segments:
            if seg.embedding is None:
                continue
            sim = self.cosine_similarity(seg.embedding, target_embedding)
            if sim >= min_similarity:
                seg.identity = "target"
                target_segments.append(seg)
                logger.debug(f"Matched segment [{seg.start:.1f}-{seg.end:.1f}] sim={sim:.3f}")
        
        if not target_segments:
            logger.info("No segments matched target speaker")
            return np.array([]), []
        
        # 3. 合并目标片段
        # 在片段之间插入静音
        silence_duration = 0.1  # 100ms 静音
        silence = np.zeros(int(silence_duration * sr))
        
        parts = []
        for i, seg in enumerate(target_segments):
            if i > 0:
                parts.append(silence)
            parts.append(seg.audio)
        
        merged_audio = np.concatenate(parts)
        
        total_speech = sum(s.end - s.start for s in target_segments)
        logger.info(f"Extracted target speaker: {total_speech:.1f}s speech from {len(audio)/sr:.1f}s audio")
        
        return merged_audio, target_segments
    
    # ========== 工具方法 ==========
    
    def get_speaker_count_estimate(self, audio: np.ndarray, sr: int = 16000) -> int:
        """估计音频中说话人数量"""
        segments = self.diarize(audio, sr)
        if not segments:
            return 0
        return len(set(s.speaker_idx for s in segments))
    
    def audio_stats(self, audio: np.ndarray, sr: int) -> Dict:
        """音频统计信息"""
        return {
            "duration": float(len(audio) / sr),
            "sample_rate": sr,
            "samples": len(audio),
            "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
            "peak": float(np.max(np.abs(audio))),
        }
