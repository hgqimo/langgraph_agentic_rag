"""LangGraph Agentic RAG：检索工具（R 模块）。

对应 qa_manager/base_agent_tag.py::RetrievalTool：非 LLM 工具，
POST {"questions": [q], "N": k} 到检索服务，取返回体里 data[0]["top_k_docs"]，
URL 池随机选择 + 失败重试。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import requests

logger = logging.getLogger(__name__)

_TEXT_KEYS = ("text", "content", "document", "passage", "body", "chunk", "title")


def normalize_doc(doc: Any) -> str:
    """检索服务返回的文档可能是 str / dict，统一成 prompt 里可用的文本。"""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in _TEXT_KEYS:
            if doc.get(key):
                return str(doc[key])
        return json.dumps(doc, ensure_ascii=False)
    return str(doc)


@dataclass
class RetrievalTool:
    api_urls: List[str] = field(default_factory=lambda: ["http://127.0.0.1:8000/search"])
    timeout: float = 5.0
    max_attempts: int = 5
    mock: bool = False
    raise_on_failure: bool = False

    def query(self, question: str, n: int = 5) -> List[str]:
        """返回 n 篇最相关文档的文本列表。"""
        if self.mock:
            return self._mock_docs(question, n)

        payload = {"questions": [question], "N": n}
        headers = {"Content-Type": "application/json"}
        tried: set = set()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            candidates = [u for u in self.api_urls if u not in tried] or list(self.api_urls)
            api_url = random.choice(candidates)
            tried.add(api_url)
            try:
                resp = requests.post(
                    api_url, headers=headers, data=json.dumps(payload), timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                docs = self._parse_response(data)
                return [normalize_doc(d) for d in docs][:n]
            except Exception as exc:  # noqa: BLE001 - 与原始实现一致：任何异常都换 URL 重试
                last_error = exc
                logger.warning("检索第 %d 次尝试失败（%s）：%s", attempt, api_url, exc)

        message = f"检索失败（已尝试 {self.max_attempts} 次）：{last_error}"
        if self.raise_on_failure:
            raise RuntimeError(message)
        logger.error(message)
        return []

    @staticmethod
    def _parse_response(data: Any) -> List[Any]:
        """兼容 data[0]["top_k_docs"] / {"top_k_docs": ...} / list 三种形态。"""
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "top_k_docs" in first:
                docs = first["top_k_docs"]
                # 兼容 [[{...}, {...}]] 这种嵌套
                if docs and isinstance(docs[0], list):
                    docs = docs[0]
                return list(docs)
            return list(data)
        if isinstance(data, dict) and "top_k_docs" in data:
            return list(data["top_k_docs"])
        raise ValueError("Unexpected API response format")

    @staticmethod
    def _mock_docs(question: str, n: int) -> List[str]:
        # 注意：mock 文档不内联原始 question，避免下游 Agent prompt 被污染
        return [
            f"[mock document {i}] Mock retrieved passage containing the evidence needed to answer the question."
            for i in range(max(0, n))
        ]
