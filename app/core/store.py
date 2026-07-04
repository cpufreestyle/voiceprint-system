"""
VoicePrint Store - 声纹库管理
SQLite 存储 + 内存缓存
"""

import os
import json
import sqlite3
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VoicePrintStore:
    """声纹库 - SQLite 持久化 + 内存缓存"""
    
    def __init__(self, db_path: str = "./data/voiceprint.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, np.ndarray] = {}
        self._init_db()
        self._load_cache()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sample_count INTEGER DEFAULT 1,
                active INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verify_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                success INTEGER,
                score REAL,
                timestamp TEXT,
                audio_duration REAL,
                speaker_count INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_cache(self):
        """加载所有声纹到内存"""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT user_id, embedding, embedding_dim FROM speakers WHERE active=1").fetchall()
        conn.close()
        
        for user_id, emb_blob, dim in rows:
            self._cache[user_id] = np.frombuffer(emb_blob, dtype=np.float32).reshape(dim)
        
        logger.info(f"Loaded {len(self._cache)} speaker embeddings into cache")
    
    def enroll(self, user_id: str, name: str, embedding: np.ndarray) -> bool:
        """注册或更新用户声纹"""
        now = datetime.now().isoformat()
        emb_bytes = embedding.astype(np.float32).tobytes()
        dim = embedding.shape[0]
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            # 如果已存在，做平均融合（多段注册提升准确率）
            existing = conn.execute(
                "SELECT embedding, sample_count FROM speakers WHERE user_id=?", (user_id,)
            ).fetchone()
            
            if existing:
                old_emb = np.frombuffer(existing[0], dtype=np.float32).reshape(dim)
                count = existing[1]
                # 加权平均：新声纹权重 0.4，旧声纹权重 0.6
                merged = 0.6 * old_emb + 0.4 * embedding
                merged = merged / np.linalg.norm(merged)  # 归一化
                emb_bytes = merged.astype(np.float32).tobytes()
                count += 1
                
                conn.execute(
                    "UPDATE speakers SET embedding=?, updated_at=?, sample_count=? WHERE user_id=?",
                    (emb_bytes, now, count, user_id)
                )
                self._cache[user_id] = merged
                logger.info(f"Updated speaker '{user_id}' (sample #{count})")
            else:
                conn.execute(
                    "INSERT INTO speakers (user_id, name, embedding, embedding_dim, created_at, updated_at, sample_count, active) VALUES (?,?,?,?,?,?,?,1)",
                    (user_id, name, emb_bytes, dim, now, now, 1)
                )
                self._cache[user_id] = embedding
                logger.info(f"Enrolled new speaker '{user_id}' ({name})")
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Enroll failed: {e}")
            return False
        finally:
            conn.close()
    
    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """获取用户声纹向量"""
        return self._cache.get(user_id)
    
    def list_speakers(self) -> List[Dict]:
        """列出所有已注册用户"""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT user_id, name, created_at, updated_at, sample_count, active FROM speakers ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "user_id": r[0],
                "name": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "sample_count": r[4],
                "active": bool(r[5]),
            }
            for r in rows
        ]
    
    def delete_speaker(self, user_id: str) -> bool:
        """删除用户"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.execute("DELETE FROM speakers WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        
        if user_id in self._cache:
            del self._cache[user_id]
        
        deleted = cur.rowcount > 0
        if deleted:
            logger.info(f"Deleted speaker '{user_id}'")
        return deleted
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """获取所有声纹（用于 1:N 识别）"""
        return [(uid, emb) for uid, emb in self._cache.items()]
    
    def log_verify(self, user_id: str, success: bool, score: float,
                   audio_duration: float = 0, speaker_count: int = 1):
        """记录验证日志"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO verify_logs (user_id, success, score, timestamp, audio_duration, speaker_count) VALUES (?,?,?,?,?,?)",
            (user_id, int(success), score, datetime.now().isoformat(), audio_duration, speaker_count)
        )
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """统计信息"""
        conn = sqlite3.connect(str(self.db_path))
        total = conn.execute("SELECT COUNT(*) FROM speakers WHERE active=1").fetchone()[0]
        total_logs = conn.execute("SELECT COUNT(*) FROM verify_logs").fetchone()[0]
        success_logs = conn.execute("SELECT COUNT(*) FROM verify_logs WHERE success=1").fetchone()[0]
        conn.close()
        
        return {
            "total_speakers": total,
            "total_verifications": total_logs,
            "success_rate": success_logs / total_logs if total_logs > 0 else 0,
        }
