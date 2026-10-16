"""LangGraph Agentic RAG：运行时。

把「共享骨干 LLM + 检索工具 + 配置」打包成 Runtime，供各节点闭包调用；
Runtime.complete 对应原仓库 BaseLLMClient.async_get_response 的调用语义
（按 Agent 角色设定温度、失败重试）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import convert_to_messages

from .config import RagConfig
from .llm import text_of
from .tools import RetrievalTool

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    cfg: RagConfig
    llm: BaseChatModel
    # 可注入任意实现 query(question, n) -> List[str] 的检索器（如 VectorRetriever）；
    # 为 None 时按 cfg.retrieval_mode 自动构造。
    retriever: Any = None
    # GAM-RAG 风格经验库（ExperienceStore）；None 且 cfg.memory_feedback=off 时不启用
    memory: Any = None
    call_count: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.retriever is None:
            mode = self.cfg.resolved_retrieval_mode()
            if mode == "vector":
                # 惰性导入：只有走向量检索时才需要 numpy
                from .vector import build_retriever_from_config

                self.retriever = build_retriever_from_config(self.cfg)
            elif mode == "engine":
                # 惰性导入：Corpus Interaction Engine（向量 + FTS5 融合）
                from .engine import build_engine_from_config

                self.retriever = build_engine_from_config(self.cfg)
            else:
                self.retriever = RetrievalTool(
                    api_urls=self.cfg.search_urls,
                    timeout=self.cfg.retrieval_timeout,
                    max_attempts=self.cfg.retrieval_max_attempts,
                    mock=mode == "mock",
                    raise_on_failure=self.cfg.retrieval_raise,
                )
        self._init_memory()
        # Jev 决策模型重排：必须在记忆挂载之后包装（memory 直接挂到内层 engine；
        # 装饰器 __getattr__ 透明转发 memory/semantic/docs 等属性）
        if str(getattr(self.cfg, "rerank_mode", "off") or "off").lower() not in ("", "off"):
            from .rerank import wrap_retriever

            self.retriever = wrap_retriever(self.retriever, self.cfg)

    # ------------------------------------------------------------------ 自进化记忆
    def _semantic_retriever(self) -> Any:
        """返回能提供 embedding 矩阵的底层 VectorRetriever（引擎则取其 semantic）。"""
        retriever = self.retriever
        if hasattr(retriever, "semantic"):
            return getattr(retriever, "semantic")
        if hasattr(retriever, "matrix") and hasattr(retriever, "embed_query"):
            return retriever
        return None

    def _init_memory(self) -> None:
        mode = str(self.cfg.memory_feedback or "off").lower()
        if self.memory is not None:
            # 外部显式注入的 store 也要挂到引擎上
            if hasattr(self.retriever, "memory"):
                self.retriever.memory = self.memory
            return
        if mode in ("", "off"):
            return
        # 记忆依赖「稳定语料 id + search(plan) 门控」，只有引擎模式两者兼备；
        # 纯向量/HTTP 模式下 doc id 是列表位置（不稳定），写回会污染记忆，直接禁用。
        if not hasattr(self.retriever, "memory"):
            logger.warning(
                "memory_feedback=%s 需要 retrieval_mode=engine（当前检索器无稳定 doc id/门控），已禁用",
                mode,
            )
            return
        semantic = self._semantic_retriever()
        if semantic is None or getattr(semantic, "matrix", None) is None:
            logger.warning("memory_feedback=%s 需要向量索引，已禁用记忆", mode)
            return
        from .memory import ExperienceStore

        cfg = self.cfg
        self.memory = ExperienceStore(
            cfg.memory_path,
            r_pos=cfg.memory_r_pos,
            r_neg=cfg.memory_r_neg,
            q_noise=cfg.memory_q_noise,
            exclude_cos=cfg.memory_exclude_cos,
            exclude_pi=cfg.memory_exclude_pi,
            inject_max=cfg.memory_inject_max,
            inject_cos=cfg.memory_inject_cos,
            inject_pi=cfg.memory_inject_pi,
            strategy_max_entries=getattr(cfg, "memory_strategy_max_entries", 64),
            strategy_min_pos=getattr(cfg, "memory_strategy_min_pos", 3),
        )
        self.retriever.memory = self.memory
        logger.info("自进化记忆已启用：feedback=%s db=%s", mode, cfg.memory_path or "<内存>")

    def doc_text(self, doc_id: int) -> str:
        """按语料 id 取段落全文（记忆反馈的标注以全文为准，而非句子片段）。"""
        docs = getattr(self.retriever, "docs", None)
        if docs is None:
            semantic = self._semantic_retriever()
            docs = getattr(semantic, "docs", None)
        if docs and 0 <= int(doc_id) < len(docs):
            return str(docs[int(doc_id)])
        return ""

    def apply_memory_feedback(self, state: Dict[str, Any]) -> Dict[str, float]:
        """轨迹结束后把本次 rollout 的检索情节写回记忆库（GAM-RAG §3.3 的在线更新）。

        - gold 模式（离线评测）：检回段落含金标 evidence → y=+1，其余 → y=-1（R_neg 保守）
        - agent 模式（生产）：judge 明确排除 → y=-1；最终保留进 AG 上下文 → y=+1；其余跳过
        每个 (检索 query, 段落) 是一次独立情节，沿各自 query 方向做 Kalman 修正。
        """
        mode = str(self.cfg.memory_feedback or "off").lower()
        semantic = self._semantic_retriever()
        if self.memory is None or mode in ("", "off"):
            return {}
        if semantic is None or getattr(semantic, "matrix", None) is None:
            return {}

        def norm(text: str) -> str:
            return " ".join(str(text).split()).lower()

        gold = [norm(e) for e in (state.get("gold_evidence") or []) if e and str(e).strip()]
        if mode == "gold" and not gold:
            return {}

        episodes: List[tuple] = []  # (query_text, doc_id)
        judge_excluded: set = set()
        for step in state.get("trace") or []:
            agent = step.get("agent")
            extra = step.get("extra") or {}
            if agent == "RetrievalTool":
                messages = step.get("messages") or []
                qtext = str(messages[0].get("content", "")) if messages else ""
                for doc_id in extra.get("doc_ids") or []:
                    episodes.append((qtext, int(doc_id)))
            elif agent == "EvidenceJudgeAgent":
                judge_excluded |= {int(i) for i in extra.get("exclude") or []}
        if not episodes:
            return {}

        kept_texts: List[str] = [norm(t) for t in state.get("docs") or []]
        for sub in state.get("sub_queries") or []:
            kept_texts.extend(norm(t) for t in (sub.get("docs") or []))
        kept_texts = [t for t in kept_texts if t]

        labels: Dict[tuple, float] = {}
        for qtext, doc_id in episodes:
            full = norm(self.doc_text(doc_id))
            if not qtext or not full:
                continue
            if mode == "gold":
                labels[(qtext, doc_id)] = 1.0 if any(g in full for g in gold) else -1.0
            elif doc_id in judge_excluded:
                labels[(qtext, doc_id)] = -1.0
            elif any(t in full or full in t for t in kept_texts):
                labels[(qtext, doc_id)] = 1.0

        updated = {"pos": 0.0, "neg": 0.0}
        matrix = semantic.matrix
        for (qtext, doc_id), y in labels.items():
            if not (0 <= doc_id < len(matrix)):
                continue
            try:
                qvec = semantic.embed_query(qtext)
            except Exception as exc:  # noqa: BLE001
                logger.warning("记忆反馈编码失败（跳过）：%s", exc)
                continue
            self.memory.feedback(doc_id, matrix[doc_id], qvec, y)
            updated["pos" if y > 0 else "neg"] += 1.0

        # 策略记忆（P1/P2.1）：把「带来成功检索的 plan」写回策略库。
        # gold 口径按轮归因：本轮检回的段落命中金标 → 该轮 plan y=+1，否则 y=-1；
        # agent 口径（生产）收紧（P2.1a）：judge 充足 且 答案 grounded（非拒答、
        # 与 AG 证据文本有重叠）才算成功——musique 案例证明仅 judge sufficient
        # 太松，低质策略滚雪球入库导致 warm-start 污染（hitR 1.0→0.95）。
        if getattr(self.cfg, "memory_strategies", True) and hasattr(self.memory, "update_strategy"):
            updated["strategy_pos"] = 0.0
            updated["strategy_neg"] = 0.0
            last_judge_ok: Any = None
            for step in state.get("trace") or []:
                extra = step.get("extra") or {}
                if step.get("agent") == "EvidenceJudgeAgent" and "sufficient" in extra:
                    last_judge_ok = bool(extra.get("sufficient"))
            grounded = self._answer_grounded(state, kept_texts)
            embed_cache: Dict[str, Any] = {}
            for step in state.get("trace") or []:
                if step.get("agent") != "RetrievalTool":
                    continue
                extra = step.get("extra") or {}
                plan = {
                    k: v for k, v in (extra.get("plan") or {}).items()
                    if k in ("keywords_must", "keywords_any", "keywords", "entity", "weights", "include") and v
                }
                if not plan:
                    continue  # 基础检索轮（无策略）不入策略库
                messages = step.get("messages") or []
                qtext = str(messages[0].get("content", "")) if messages else ""
                if not qtext:
                    continue
                if mode == "gold":
                    y = 1.0 if any(
                        g in norm(self.doc_text(int(d)))
                        for d in (extra.get("doc_ids") or [])
                        for g in gold
                    ) else -1.0
                else:
                    if last_judge_ok is None:
                        continue
                    y = 1.0 if (last_judge_ok and grounded) else -1.0
                if qtext not in embed_cache:
                    try:
                        embed_cache[qtext] = semantic.embed_query(qtext)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("策略反馈编码失败（跳过）：%s", exc)
                        embed_cache[qtext] = None
                if embed_cache[qtext] is None:
                    continue
                self.memory.update_strategy(qtext, embed_cache[qtext], plan, y)
                updated["strategy_pos" if y > 0 else "strategy_neg"] += 1.0
        return updated

    @staticmethod
    def _answer_grounded(state: Dict[str, Any], kept_texts: List[str]) -> bool:
        """P2.1a 自洽校验：最终答案非拒答，且与 AG 证据文本有实质重叠。

        生产可用的强信号（无需 GT）：答案在证据里「有着落」说明检索→聚焦链路
        自洽；拒答或证据外答案视为该轨迹的策略不可信。
        """
        pred = str(
            state.get("answer") or state.get("pred") or state.get("pred_answer") or ""
        ).strip()
        if not pred:
            return False
        low = " ".join(pred.split()).lower()
        if low.startswith(("i don't know", "i do not know", "unknown", "insufficient", "no answer")):
            return False
        # 短答案整体匹配；长答案取前 40 字符（避免生成式扩写导致失配）
        probe = low if len(low) <= 40 else low[:40]
        return any(probe in t for t in kept_texts if t)

    def lookup_strategy(self, query_text: str) -> Any:
        """P1 warm-start：按子问题语义键查策略库。

        返回 (key, plan, cos) 或 None；仅 memory_strategies 开启且记忆库可用时生效。
        """
        if self.memory is None or not hasattr(self.memory, "lookup_strategy"):
            return None
        if not getattr(self.cfg, "memory_strategies", True):
            return None
        semantic = self._semantic_retriever()
        if semantic is None:
            return None
        try:
            qvec = semantic.embed_query(query_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("策略查询编码失败（跳过 warm-start）：%s", exc)
            return None
        return self.memory.lookup_strategy(qvec)

    def temperature_for(self, agent_name: str) -> float:
        return self.cfg.agent_temperatures.get(agent_name, self.cfg.default_temperature)

    def retrieve(self, question: str, k: int) -> List[str]:
        """R 模块入口：按 retrieval_mode 分派 http / mock / none / llm。"""
        return [d.text for d in self.retrieve_docs({"semantic_query": question, "top_k": k})]

    def retrieve_docs(self, plan: Dict[str, Any]) -> List[Any]:
        """R 模块入口（带动作原语的版本）。

        - 检索器实现了 search(plan)（即 Corpus Interaction Engine）时，走多面检索 + 上下文整形；
        - 否则退化为纯 query 检索（keywords/entity/include/exclude 会被忽略）。
        """
        mode = self.cfg.resolved_retrieval_mode()
        if mode == "none":
            return []
        if hasattr(self.retriever, "search") and mode in ("engine", "vector"):
            return list(self.retriever.search(plan))

        from .engine import RetrievedDoc

        query = str(plan.get("semantic_query") or "")
        k = int(plan.get("top_k") or self.cfg.top_k)
        if mode == "llm":
            texts = self._llm_retrieve(query, k)
        else:
            texts = self.retriever.query(query, k)
        return [RetrievedDoc(id=i, text=t, score=0.0) for i, t in enumerate(texts)]

    def _llm_retrieve(self, question: str, k: int) -> List[str]:
        """无检索服务时的降级方案：让本地模型按查询自造 k 篇候选证据文档。

        注意：这不是真实检索，仅用于在没有搜索引擎时验证 R→DS→AG 的完整拓扑
        与 prompt/解析链路。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the backend of a web search engine. Given a query, output at most "
                    f"{k} short candidate documents that a search engine would return for it. "
                    "One document per line, each line starting with '- ' and being a self-contained "
                    "factual passage (20-60 words). Include concrete entities, dates and numbers. "
                    "Do not answer the query, do not add headings, do not explain."
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            text = self.complete("RetrievalTool", messages)
        except RuntimeError as exc:
            logger.error("LLM 检索失败：%s", exc)
            return []
        docs = [line.strip().lstrip("-*0123456789. ").strip() for line in text.splitlines()]
        return [d for d in docs if d][:k]

    def complete(self, agent_name: str, messages: List[Dict[str, str]]) -> str:
        """按角色温度调用共享骨干，带重试（对应原仓库 max_retries / retry_interval）。"""
        self.call_count[agent_name] = self.call_count.get(agent_name, 0) + 1
        bound = self.llm.bind(temperature=self.temperature_for(agent_name))
        last_error: Exception | None = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                response = bound.invoke(convert_to_messages(messages))
                return text_of(response)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("%s 第 %d 次调用失败：%s", agent_name, attempt, exc)
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_interval)

        raise RuntimeError(f"{agent_name} 调用失败：{last_error}")
