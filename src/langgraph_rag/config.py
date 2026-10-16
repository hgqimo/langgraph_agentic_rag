"""LangGraph Agentic RAG：全局配置。

默认值与 上游实现保持一致：
- 各 Agent 温度：qa_manager/base_agent_tag.py 中各 Agent 的 default_temperature
- penalty / workflow 回退 / turn 统计：verl/trainer/ppo/ray_trainer.py::async_rollout_single
- 检索服务地址池：qa_manager/qa.py 中的 ips + 8 个端口
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

# 与原仓库 record_metrics / single_batch_list 对应的可训练 Agent
# （最后一项 EvidenceJudgeAgent 对应 INTERACT-RAG 的 Adaptive-Reasoner，
#   用于运行期「检索→判断→不足→重写→再检索」循环，原仓库没有）
AGENT_NAMES: List[str] = [
    "PlanningAgent",
    "SubPlanningAgent",
    "QueryRewriteAgent",
    "QueryDecompositionAgentParallel",
    "QueryDecompositionAgentSerial",
    "DocumentSelectionAgent",
    "AnswerGenerationAgent",
    "AnswerSummarizationAgent",
    "EvidenceJudgeAgent",
]

DEFAULT_TEMPERATURES: Dict[str, float] = {
    "PlanningAgent": 0.5,
    "SubPlanningAgent": 0.5,
    "QueryRewriteAgent": 0.5,
    "QueryDecompositionAgentParallel": 0.3,
    "QueryDecompositionAgentSerial": 0.3,
    "DocumentSelectionAgent": 0.0,
    "AnswerGenerationAgent": 0.0,
    "AnswerSummarizationAgent": 0.0,
    "EvidenceJudgeAgent": 0.0,
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _default_search_urls() -> List[str]:
    """对齐 qa_manager/qa.py: ips=['10.148.12.65'], 端口 8000~8007 的 /search。"""
    raw = os.getenv("RAG_SEARCH_URLS")
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    host = os.getenv("RAG_SEARCH_HOST", "127.0.0.1")
    return [f"http://{host}:{8000 + i}/search" for i in range(8)]


@dataclass
class RagConfig:
    # ---------------- 模型（单一共享骨干，对应 上游的 shared backbone） ----------------
    model: str = field(default_factory=lambda: _env("RAG_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    base_url: str = field(default_factory=lambda: _env("RAG_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key: str = field(default_factory=lambda: _env("RAG_API_KEY", "EMPTY"))
    default_temperature: float = field(default_factory=lambda: _env_float("RAG_TEMPERATURE", 0.6))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_MAX_TOKENS", 512))
    max_retries: int = field(default_factory=lambda: _env_int("RAG_MAX_RETRIES", 3))
    retry_interval: float = field(default_factory=lambda: _env_float("RAG_RETRY_INTERVAL", 1.0))
    agent_temperatures: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TEMPERATURES))

    # 离线 mock：无需真实 LLM 即可跑通整张图（验证拓扑、分支、指标）
    mock: bool = field(default_factory=lambda: _env_bool("RAG_MOCK", False))
    mock_route: str = field(default_factory=lambda: _env("RAG_MOCK_ROUTE", "qds"))

    # ---------------- 模型调用扩展 ----------------
    # Qwen3 系列（vLLM --reasoning-parser qwen3）默认开启思考模式，需显式关闭：
    #   {"chat_template_kwargs": {"enable_thinking": false}}
    # 其他后端如需额外请求体参数，可用 RAG_EXTRA_BODY 传 JSON（会与下面的设置合并）。
    disable_thinking: bool = field(default_factory=lambda: _env_bool("RAG_DISABLE_THINKING", True))
    extra_body_json: str = field(default_factory=lambda: _env("RAG_EXTRA_BODY", ""))

    # ---------------- 检索工具（R 模块，非 LLM） ----------------
    # auto: mock 模式用假文档，否则走 HTTP 检索服务
    # http / mock / none(不检索，链中自动去掉 R、DS)
    # llm(用本地模型自造候选证据) / vector(本地纯向量索引)
    # engine(Corpus Interaction Engine：向量+BM25 融合+entity+include/exclude)
    retrieval_mode: str = field(default_factory=lambda: _env("RAG_RETRIEVAL_MODE", "auto"))
    search_urls: List[str] = field(default_factory=_default_search_urls)

    # ---------------- 向量检索（BGE 等 OpenAI 兼容 embeddings 服务） ----------------
    embed_base_url: str = field(default_factory=lambda: _env("RAG_EMBED_BASE_URL", "http://127.0.0.1:10200/v1"))
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "text-embedding-3-small"))
    embed_api_key: str = field(default_factory=lambda: _env("RAG_EMBED_API_KEY", "EMPTY"))
    embed_batch_size: int = field(default_factory=lambda: _env_int("RAG_EMBED_BATCH", 16))
    corpus_path: Optional[str] = field(default_factory=lambda: _env("RAG_CORPUS", "") or None)
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    retrieval_timeout: float = field(default_factory=lambda: _env_float("RAG_RETRIEVAL_TIMEOUT", 5.0))
    retrieval_max_attempts: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_ATTEMPTS", 5))
    retrieval_raise: bool = field(default_factory=lambda: _env_bool("RAG_RETRIEVAL_RAISE", False))

    # ---------------- Jev 决策模型重排（NeoHorse-Jev-4B，HTTP 服务） ----------------
    # off = 关闭（默认；端到端对多轮 agent 中性，见 rerank.py 验证结论）
    # single_pass = 一次 Choice forward 重排整个候选池（检索级 musique hit@8 +2.7pt）
    # tournament  = BlitzRank 图排序（效果≈single_pass，~5x 调用成本）
    rerank_mode: str = field(default_factory=lambda: _env("RAG_RERANK_MODE", "off"))
    # qwen3 = Qwen3-Reranker-4B cross-encoder（默认；300 题对比 musique hit@8
    # 0.9867 vs Jev 0.9600，速度同量级）; jev = NeoHorse-Jev-4B 决策模型
    rerank_backend: str = field(default_factory=lambda: _env("RAG_RERANK_BACKEND", "qwen3"))
    rerank_url: str = field(default_factory=lambda: _env("RAG_RERANK_URL", ""))
    rerank_pool: int = field(default_factory=lambda: _env_int("RAG_RERANK_POOL", 30))
    rerank_max_doc_chars: int = field(default_factory=lambda: _env_int("RAG_RERANK_MAX_DOC_CHARS", 600))

    # ---------------- 运行期检索循环（INTERACT-RAG 的 Reasoner 回路） ----------------
    # 1 = 现行为（单次检索，不调用 judge）；>1 = 启用「检索→判断→不足→重写→再检索」
    # PlanningAgent 不感知该开关：plan 阶段只决定形状（QR/R/DS/AG 或 QDS/QDP），
    # 是否要再检索一轮由运行期的 EvidenceJudgeAgent 决定。
    max_retrieval_rounds: int = field(default_factory=lambda: _env_int("RAG_MAX_ROUNDS", 1))
    # 循环内重写 query 时允许注入的「还缺什么」上下文长度上限（词）
    judge_missing_max_words: int = field(default_factory=lambda: _env_int("RAG_JUDGE_MISSING_WORDS", 60))

    # ---------------- 自进化记忆（GAM-RAG 风格段落记忆，training-free） ----------------
    # off  = 关闭（默认，行为与无记忆版一致）
    # gold = 离线评测口径：检回且含金标 evidence 的段落 y=+1，其余 y=-1
    # agent= 生产口径：judge 明确排除的段落 y=-1，最终保留（DS/AG 上下文）的 y=+1，其余跳过
    # 仅在 retrieval_mode=engine 时生效（需要段落 embedding 与稳定 doc id）。
    memory_feedback: str = field(default_factory=lambda: _env("RAG_MEMORY_FEEDBACK", "off"))
    # SQLite 持久化路径（跨进程进化）；空 = 纯进程内存（多轮评测够用）
    memory_path: Optional[str] = field(default_factory=lambda: _env("RAG_MEMORY_DB", "") or None)
    # Kalman 超参（默认对齐论文 Appendix C：R_pos=0.5, R_neg=1.0 非对称保守）
    memory_r_pos: float = field(default_factory=lambda: _env_float("RAG_MEMORY_R_POS", 0.5))
    memory_r_neg: float = field(default_factory=lambda: _env_float("RAG_MEMORY_R_NEG", 1.0))
    memory_q_noise: float = field(default_factory=lambda: _env_float("RAG_MEMORY_Q", 0.05))
    # 持久排除 / 记忆激活注入阈值
    memory_exclude_cos: float = field(default_factory=lambda: _env_float("RAG_MEMORY_EXCLUDE_COS", -0.25))
    memory_exclude_pi: float = field(default_factory=lambda: _env_float("RAG_MEMORY_EXCLUDE_PI", 0.35))
    memory_inject_max: int = field(default_factory=lambda: _env_int("RAG_MEMORY_INJECT", 2))
    memory_inject_cos: float = field(default_factory=lambda: _env_float("RAG_MEMORY_INJECT_COS", 0.65))
    memory_inject_pi: float = field(default_factory=lambda: _env_float("RAG_MEMORY_INJECT_PI", 0.5))
    # P1 策略记忆：semantic_key→judge plan 的 warm-start（消融时置 False）
    memory_strategies: bool = field(
        default_factory=lambda: _env("RAG_MEMORY_STRATEGIES", "1").lower() not in ("0", "false", "no", "")
    )
    # P2.1b：策略库容量上限（超限按支持度 s 低者优先淘汰；0=不限）
    memory_strategy_max_entries: int = field(
        default_factory=lambda: _env_int("RAG_MEMORY_STRATEGY_MAX", 64)
    )
    # P3.1：warm-start 要求的最小独立成功次数（消融证明 2 次自我标注成功
    # 不足以保 plan 质量，musique agent 臂 ws 污染 hitR 1.00→0.97）
    memory_strategy_min_pos: int = field(
        default_factory=lambda: _env_int("RAG_MEMORY_STRATEGY_MIN_POS", 3)
    )

    # ---------------- 流程开关（对齐 async_rollout_single） ----------------
    enable_qdp: bool = field(default_factory=lambda: _env_bool("RAG_ENABLE_QDP", True))
    enable_doc_selection: bool = field(default_factory=lambda: _env_bool("RAG_ENABLE_DS", True))
    # 原仓库中 SubPlanningAgent 的输出被硬编码覆盖（"no training planner + trained executors" 对照实验）
    sub_workflow_override: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_SUB_WORKFLOW_OVERRIDE", "R,DS,AG") or None
    )
    invalid_penalty: float = field(default_factory=lambda: _env_float("RAG_INVALID_PENALTY", -2.0))
    default_workflow_on_invalid: str = field(
        default_factory=lambda: _env("RAG_DEFAULT_WORKFLOW", "QDS")
    )
    max_sub_questions: int = field(default_factory=lambda: _env_int("RAG_MAX_SUB_QUESTIONS", 7))
    # 成本惩罚系数（原仓库当前 alpha=beta=0，即未启用）
    alpha_turn_penalty: float = field(default_factory=lambda: _env_float("RAG_ALPHA", 0.0))
    beta_retrieval_penalty: float = field(default_factory=lambda: _env_float("RAG_BETA", 0.0))

    def with_overrides(self, **kwargs) -> "RagConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def resolved_retrieval_mode(self) -> str:
        if self.retrieval_mode == "auto":
            return "mock" if self.mock else "http"
        return self.retrieval_mode

    def build_extra_body(self) -> Dict[str, Any]:
        """合并 RAG_EXTRA_BODY 与「关闭思考模式」参数，作为请求体附加字段。"""
        body: Dict[str, Any] = json.loads(self.extra_body_json) if self.extra_body_json else {}
        if self.disable_thinking:
            chat_kwargs = dict(body.get("chat_template_kwargs") or {})
            chat_kwargs.setdefault("enable_thinking", False)
            body["chat_template_kwargs"] = chat_kwargs
        return body
