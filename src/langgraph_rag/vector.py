"""LangGraph Agentic RAG：向量检索（对接 OpenAI 兼容的 embeddings 服务，如本机 BGE）。

与原仓库的 HTTP 检索服务（RetrievalTool）接口一致（query(question, n)），
因此可直接替换进 R 模块，但这里是本地向量索引 + 余弦相似度检索。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class VectorRetriever:
    """内存向量检索：批量编码语料 -> L2 归一化 -> 余弦 top-k。"""

    docs: List[str]
    base_url: str = "http://127.0.0.1:10200/v1"
    model: str = "text-embedding-3-small"
    api_key: str = "EMPTY"
    batch_size: int = 16
    timeout: float = 60.0
    max_retries: int = 3
    normalize: bool = True
    cache_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_CACHE_DIR", str(Path.home() / ".cache" / "jade_langgraph"))
    )
    verbose: bool = True
    matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # query embedding 缓存：检索循环/记忆反馈会对同一 query 文本反复取向量
    _query_cache: dict = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ 编码
    def _embed(self, texts: List[str], desc: str = "") -> np.ndarray:
        """分批调用 /v1/embeddings，返回 (n, dim) 矩阵。"""
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        vectors: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": chunk, "encoding_format": "float"}
            last_error: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    # 按 index 排序，保证与输入顺序一致
                    data = sorted(data, key=lambda item: item.get("index", 0))
                    vectors.extend([item["embedding"] for item in data])
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning("embedding 第 %d 次失败：%s", attempt, exc)
                    time.sleep(1.0)
            else:
                raise RuntimeError(f"embedding 请求失败：{last_error}")

            if self.verbose and (start // self.batch_size) % 10 == 0:
                logger.info("  编码 %s %d/%d", desc, min(start + self.batch_size, len(texts)), len(texts))

        matrix = np.asarray(vectors, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix

    def _cache_file(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(
            ("\n".join(self.docs) + f"|{self.base_url}|{self.model}").encode("utf-8")
        ).hexdigest()[:16]
        return Path(self.cache_dir) / f"emb_{digest}.npy"

    def build(self) -> "VectorRetriever":
        """构建索引；命中缓存则直接复用（语料/模型/地址任一变化都会失效）。"""
        cache_file = self._cache_file()
        if cache_file and cache_file.exists():
            self.matrix = np.load(cache_file)
            if self.verbose:
                logger.info("  复用向量缓存 %s (%s)", cache_file.name, self.matrix.shape)
            return self

        self.matrix = self._embed(self.docs, desc="corpus")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self.matrix)
            if self.verbose:
                logger.info("  向量已缓存到 %s (%s)", cache_file, self.matrix.shape)
        return self

    # ------------------------------------------------------------------ 检索
    def embed_query(self, question: str) -> np.ndarray:
        """单条 query 编码（L2 归一化，带缓存）。"""
        cached = self._query_cache.get(question)
        if cached is not None:
            return cached
        vec = self._embed([question], desc="query")[0]
        self._query_cache[question] = vec
        return vec

    def query(self, question: str, n: int = 5) -> List[str]:
        return [doc for doc, _ in self.query_with_scores(question, n)]

    def query_with_ids(self, question: str, n: int = 5) -> List[tuple]:
        """返回 [(语料下标, 余弦分数)]；下标即稳定 doc id，供引擎的 include/exclude 引用。"""
        if self.matrix is None:
            self.build()
        if not self.docs:
            return []

        k = min(max(1, n), len(self.docs))
        query_vec = self.embed_query(question)
        scores = self.matrix @ query_vec
        top_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def query_with_scores(self, question: str, n: int = 5) -> List[tuple]:
        return [(self.docs[i], score) for i, score in self.query_with_ids(question, n)]

    # ------------------------------------------------------------------ 语料加载
    @classmethod
    def from_jsonl(cls, path: str, **kwargs) -> "VectorRetriever":
        """从 JSONL 载入语料：每行是字符串，或 {"text"/"content": ...}。"""
        docs: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    docs.append(line)
                    continue
                if isinstance(item, str):
                    docs.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("document")
                    if text:
                        docs.append(str(text))
        return cls(docs=docs, **kwargs)


def build_retriever_from_config(cfg) -> VectorRetriever:
    """按配置从 RAG_CORPUS 构建向量检索器（CLI 用）。"""
    if not cfg.corpus_path:
        raise RuntimeError(
            "retrieval_mode=vector 需要提供语料：设置 RAG_CORPUS=<jsonl 路径>，"
            "或改用 --retrieval-mode http/mock/none，或在代码里注入 VectorRetriever。"
        )
    return VectorRetriever.from_jsonl(
        cfg.corpus_path,
        base_url=cfg.embed_base_url,
        model=cfg.embed_model,
        api_key=cfg.embed_api_key,
        batch_size=cfg.embed_batch_size,
    ).build()
