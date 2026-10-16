"""重排层（HTTP 零依赖接入，双 backend 可切换）。

backend=qwen3（默认）: Qwen3-Reranker-4B 传统 cross-encoder——官方 LogitScore
  范式（chatml 模板 + yes/no logits softmax），query-doc 成对打分，30 docs 批量
  prefill ~0.45-0.5s。300 题对比占优（musique hit@8 0.9867 vs Jev 0.9600）。
backend=jev: NeoHorse-Jev-4B 决策模型（TypeSafe AI System-One）——prefill-only，
  state + instruction + k 个候选各编码为一条以 `Decision:` 结尾的 token 路径，
  单次 forward 并行给全部候选打分，零解码 token，输出 Choice 概率分布；一次调用
  即完成整个候选池重排（~0.42s），并支持 tournament（BlitzRank 图排序）模式。

验证结论（300 道 unseen 题检索级 + musique 150 题端到端，仓库根 `jev_rerank/results/`）：
  single_pass  musique hit@8 0.9333->0.9600 (+2.7pt), hotpotqa MRR@8 0.85->0.95,
               1 call/题 ~0.42s —— 默认推荐模式;
  tournament   (BlitzRank 图排序) 效果≈single_pass 但 ~5x 调用成本，保留可选;
  RGS          净 -28 有害（一阶段强时图探索挤掉命中），不集成。
  端到端：多轮自适应 agent 会把单轮检索增益补偿吸收（JudgeAcc 噪声内、hitR
  +0.67pt、tokens -1.6%），对多轮 agent 中性；价值在单轮管线/省 tokens/大语料。

接入（cfg.rerank_mode != "off" 时 Runtime 自动包装 retriever）：
  - query(q, n)   : 内层取 max(n, pool) -> Jev 重排 -> top-n（http/vector 路径）
  - search(plan)  : plan.top_k 扩到 pool -> Jev 重排 -> 截回 top_k（engine 路径；
                    保留原 RetrievedDoc 对象，稳定 doc id 不破坏记忆反馈）
失败安全：Jev 服务不可用时原序透传（on_error=passthrough），agent 循环无感知。
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from .jev_decisions import JevDecisionsClient
from .jev_decisions import choice as jev_choice

logger = logging.getLogger(__name__)

RERANK_URL = os.environ.get("RAG_RERANK_URL", "http://127.0.0.1:8080")
QWEN3_RERANK_URL = os.environ.get("RAG_QWEN3_RERANK_URL", "http://127.0.0.1:8090")
INSTR = "Which passage best answers the query?"
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TAG = re.compile(r"<[^>]+>")


def _labels(n: int) -> List[str]:
    if not 2 <= n <= 255:
        raise ValueError(f"Choice 需要 2-255 个候选, got {n}")
    if n <= 26:
        return list(_LABELS[:n])
    return [f"{_LABELS[i // 26 - 1]}{_LABELS[i % 26]}" for i in range(n)]


def _normalize(text: str, max_chars: Optional[int] = None) -> str:
    flat = " ".join(_TAG.sub(" ", str(text)).split())
    return flat[:max_chars] if max_chars and len(flat) > max_chars else flat


class JevReranker:
    """NeoHorse-Jev-4B HTTP reranker。mode: single_pass | tournament。"""

    name = "NeoHorse-Jev-4B"

    def __init__(self, url: str = RERANK_URL, mode: str = "single_pass",
                 model: str = "NeoHorse-Jev-4B", max_doc_chars: int = 600,
                 window: int = 10, timeout: float = 60.0, max_retries: int = 30,
                 on_error: str = "passthrough"):
        if mode not in ("single_pass", "tournament"):
            raise ValueError(f"未知 rerank mode: {mode}")
        self.url = url.rstrip("/")
        self.mode = mode
        self.model = model
        self.max_doc_chars = max_doc_chars
        self.window = window
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_error = on_error
        self.forward_calls = 0
        self.candidate_tokens = 0
        self.truncated_candidates = 0
        self._ms_total = 0.0
        self.errors = 0
        # 规范化决策客户端（OpenAI Decisions API 同构，见 jev_decisions.py）
        self._client = JevDecisionsClient(url=self.url, model=model,
                                          timeout=timeout,
                                          max_retries=max_retries)

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).json().get(
                "status") == "ready"
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------ 重排
    def rerank(self, query: str, docs: Sequence[str]) -> List[str]:
        """返回重排后的文档列表（全量）。失败按 on_error 处理（默认原序透传）。"""
        docs = list(docs)
        if len(docs) < 2:
            return docs
        try:
            order = (self._single_pass(query, docs) if self.mode == "single_pass"
                     else self._tournament(query, docs))
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            if self.on_error == "raise":
                raise
            logger.warning("Jev 重排失败（原序透传）：%s", exc)
            return docs
        return [docs[i] for i in order]

    def _single_pass(self, query: str, docs: Sequence[str]) -> List[int]:
        return self._decide(query, docs)[0]

    def _tournament(self, query: str, docs: Sequence[str]) -> List[int]:
        """BlitzRank tournament graph（需要 JEV_RANKER 指向 JevRanker 源码）。"""
        jev_ranker = os.environ.get("JEV_RANKER", "")
        if not jev_ranker:
            raise RuntimeError(
                "tournament 模式需要 JEV_RANKER 指向 JevRanker 源码目录"
                "（其中需有 jev_ranker.oracle 与 jev_ranker.rerank）")
        import sys
        _stub_torch()
        if jev_ranker not in sys.path:
            sys.path.insert(0, jev_ranker)
        from jev_ranker.oracle import JevCompareOracle
        from jev_ranker.rerank import tournament_rerank
        oracle = JevCompareOracle(_PredictorShim(self), k=self.window)
        outcome = tournament_rerank(query, list(docs), oracle,
                                    top_m=self.window, max_rounds=60)
        return outcome.order

    # ------------------------------------------------------------ 决策
    def _decide(self, query: str, docs: Sequence[str]):
        """-> (order, probs, tokens, latency_ms)；一次 Choice forward，零解码。"""
        labels = _labels(len(docs))
        options = []
        for label, text in zip(labels, docs):
            flat = _normalize(text)
            if len(flat) > self.max_doc_chars:
                self.truncated_candidates += 1
            options.append({"value": label,
                            "description": flat[:self.max_doc_chars]})
        started = time.perf_counter()
        decision = self._client.decisions.create(
            input=str(query)[:4000],
            questions=[jev_choice("rank", INSTR, choices=options)])
        latency_ms = (time.perf_counter() - started) * 1000
        pmap = {p["value"]: p["probability"]
                for p in decision.answer("rank").probabilities}
        probs = [float(pmap[label]) for label in labels]
        order = sorted(range(len(docs)), key=lambda i: (-probs[i], i))
        tokens = int(decision.input_tokens)
        self.forward_calls += 1
        self.candidate_tokens += tokens
        self._ms_total += latency_ms
        return order, probs, tokens, latency_ms

    @property
    def stats(self) -> Dict[str, Any]:
        return {"forward_calls": self.forward_calls,
                "candidate_tokens": self.candidate_tokens,
                "truncated_candidates": self.truncated_candidates,
                "ms_per_call": (self._ms_total / self.forward_calls
                                if self.forward_calls else 0.0),
                "errors": self.errors}


class Qwen3Reranker:
    """传统 cross-encoder 重排（Qwen3-Reranker-4B，官方 yes/no LogitScore 服务）。

    对比验证（300 道 unseen 题，engine top-30 池 → top-8，同协议）：
      musique  hit@8 0.9867 / mrr@8 0.8463（Jev single_pass: 0.9600 / 0.7136）
      hotpotqa hit@8 0.9867 / mrr@8 0.9506（Jev: 0.9867 / 0.9511，打平）
      速度同量级（0.45-0.50s/题 vs Jev 0.42s，30 docs 批量 prefill）
    结论：专用 reranker 在难数据集（多跳）上显著更强，简单数据集打平——
    设为默认 backend；Jev 的价值在类型化决策生态（Choice/Score/Noul 复用同一服务）。
    """

    name = "Qwen3-Reranker-4B"

    def __init__(self, url: str = QWEN3_RERANK_URL, max_doc_chars: int = 600,
                 timeout: float = 300.0, max_retries: int = 3,
                 on_error: str = "passthrough"):
        self.url = url.rstrip("/")
        self.max_doc_chars = max_doc_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_error = on_error
        self.forward_calls = 0
        self.truncated_candidates = 0
        self._ms_total = 0.0
        self.errors = 0

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).json().get(
                "status") == "ready"
        except Exception:  # noqa: BLE001
            return False

    def rerank(self, query: str, docs: Sequence[str]) -> List[str]:
        """返回重排后的文档列表（全量）。失败按 on_error 处理（默认原序透传）。"""
        docs = list(docs)
        if len(docs) < 2:
            return docs
        texts = []
        for t in docs:
            flat = _normalize(t)
            if self.max_doc_chars and len(flat) > self.max_doc_chars:
                self.truncated_candidates += 1
                flat = flat[:self.max_doc_chars]
            texts.append(flat)
        try:
            order = self._decide(query, texts)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            if self.on_error == "raise":
                raise
            logger.warning("Qwen3 重排失败（原序透传）：%s", exc)
            return docs
        return [docs[i] for i in order]

    def _decide(self, query: str, texts: Sequence[str]) -> List[int]:
        started = time.perf_counter()
        payload = None
        for attempt in range(self.max_retries):
            resp = requests.post(f"{self.url}/rerank", json={
                "query": str(query)[:4000], "documents": list(texts)},
                timeout=self.timeout)
            if resp.status_code in (429, 503):
                time.sleep(1.0 + attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        if payload is None:
            raise RuntimeError(f"Qwen3 服务持续不可用 ({self.max_retries} 次)")
        self._ms_total += (time.perf_counter() - started) * 1000
        self.forward_calls += 1
        scores = payload["scores"]
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    @property
    def stats(self) -> Dict[str, Any]:
        return {"forward_calls": self.forward_calls,
                "truncated_candidates": self.truncated_candidates,
                "ms_per_call": (self._ms_total / self.forward_calls
                                if self.forward_calls else 0.0),
                "errors": self.errors}


class JevRerankRetriever:
    """检索器装饰器（对 Runtime/agent 循环完全透明）。

    - engine 路径: search(plan) —— plan.top_k 扩到 pool 取候选，Jev 重排后截回
      top_k；返回原 RetrievedDoc 对象（保留稳定 doc id，记忆反馈/门控不受影响）。
    - http/vector 路径: query(q, n) —— 内层取 max(n, pool) 再重排截 n。
    - pool <= 需求数时直通内层（不花决策调用）；其余属性经 __getattr__ 委托。
    """

    def __init__(self, inner: Any, reranker: JevReranker, pool: int = 30):
        self.inner = inner
        self.reranker = reranker
        self.pool = int(pool)

    # engine 路径
    def search(self, plan: Dict[str, Any]) -> List[Any]:
        top_k = int(plan.get("top_k") or 0) or None
        want = top_k if top_k else self.pool
        if want >= self.pool:
            return list(self.inner.search(plan))
        docs = list(self.inner.search({**plan, "top_k": self.pool}))
        if len(docs) <= want:
            return docs
        query = str(plan.get("semantic_query") or "")
        ranked = self.reranker.rerank(query, [d.text for d in docs])[:want]
        by_text: Dict[str, Any] = {}
        for d in docs:
            by_text.setdefault(d.text, d)
        out = [by_text[t] for t in ranked if t in by_text]
        return out or docs[:want]

    # http/vector 路径
    def query(self, question: str, n: int) -> List[str]:
        docs = self.inner.query(question, max(n, self.pool))
        if len(docs) <= n:
            return docs
        return self.reranker.rerank(question, docs)[:n]

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _stub_torch() -> None:
    """tournament 路径需要 import jev_ranker（其 model.py 顶层依赖 torch）。

    HTTP 决策模式下无需真 torch：stub 掉 dtype/注解/nn 符号即可（PEP 562 兜底）。
    """
    import sys
    import types
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.__version__ = "dtype-stub"

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.no_grad = _NoGrad
    torch.float32 = torch.float16 = torch.bfloat16 = "dtype-stub"
    torch.Tensor = object
    torch.device = object
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    nn.Linear = object
    nn.Parameter = lambda *a, **k: None
    nn.functional = types.SimpleNamespace(normalize=lambda *a, **k: None)
    torch.nn = nn
    torch.__getattr__ = lambda name: object
    nn.__getattr__ = lambda name: object
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn


class _PredictorShim:
    """把 JevReranker 适配成 JevRanker predictor 鸭子类型（tournament 路径用）。"""

    name = "NeoHorse-Jev-4B"

    def __init__(self, reranker: JevReranker):
        self.reranker = reranker
        self.candidate_tokens = 0

    def decide(self, state, candidates, instruction=None):
        out = self.reranker._decide(state, candidates)
        self.candidate_tokens = self.reranker.candidate_tokens
        return out

    def choice(self, state, candidates, instruction=None):
        return self.decide(state, candidates, instruction)[1]


def wrap_retriever(retriever: Any, cfg: Any) -> Any:
    """按 cfg.rerank_mode/rerank_backend 包装检索器；off/服务不健康时原样返回。

    backend: qwen3 = Qwen3-Reranker-4B cross-encoder（默认，效果对比占优）；
             jev   = NeoHorse-Jev-4B 决策模型（single_pass/tournament）。
    """
    mode = str(getattr(cfg, "rerank_mode", "off") or "off").lower()
    if mode in ("", "off"):
        return retriever
    backend = str(getattr(cfg, "rerank_backend", "qwen3") or "qwen3").lower()
    max_chars = getattr(cfg, "rerank_max_doc_chars", 600)
    if backend == "jev":
        reranker: Any = JevReranker(
            url=getattr(cfg, "rerank_url", "") or RERANK_URL,
            mode=mode, max_doc_chars=max_chars)
    elif backend == "qwen3":
        if mode == "tournament":
            logger.warning("tournament 模式仅 Jev backend 支持，回退 single_pass")
        reranker = Qwen3Reranker(
            url=getattr(cfg, "rerank_url", "") or QWEN3_RERANK_URL,
            max_doc_chars=max_chars)
    else:
        raise ValueError(f"未知 rerank_backend: {backend}")
    if not reranker.healthy():
        logger.warning("rerank_mode=%s 但 %s 服务不可用（%s），已跳过重排",
                       mode, backend, reranker.url)
        return retriever
    logger.info("重排已启用: backend=%s mode=%s pool=%s url=%s", backend, mode,
                getattr(cfg, "rerank_pool", 30), reranker.url)
    return JevRerankRetriever(retriever, reranker,
                              pool=getattr(cfg, "rerank_pool", 30))
