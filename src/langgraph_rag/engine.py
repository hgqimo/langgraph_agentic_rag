"""LangGraph Agentic RAG：Corpus Interaction Engine（对齐 INTERACT-RAG, ICLR 2026）。

把「黑盒向量检索」拆成一组可被 Agent 精细控制的动作原语（论文 §3.1）：

1) Multi-Faceted Retrieval
   - semantic search : 复用 VectorRetriever（稠密余弦）
   - exact search    : SQLite FTS5 / BM25（零新依赖，对标论文 D.2）
   - weighted fusion : 两路 top-N 各自 min-max 归一化后按权重相加
2) Anchored Matching
   - entity match    : FTS5 句子级检索，返回包含实体词的短句（段内切句）
3) Context Shaping
   - include docs / exclude docs / adjust scale(top_k)

接口对现有链路是「可丢弃替换」：既提供论文式的 search(plan) -> List[RetrievedDoc]，
也保留 query(q, n) -> List[str] 垫片，让既有 R 模块与 DS/AG 不用改。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .vector import VectorRetriever

logger = logging.getLogger(__name__)

SEMANTIC = "semantic"
EXACT = "exact"
FUSION = "fusion"
ENTITY = "entity"
MEMORY = "memory"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# 无显式关键词时，用 query 自身的高信息量词做词法分支（去掉高频虚词）
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "that", "this", "these", "those", "it", "its", "as", "than", "then", "there",
    "their", "them", "they", "he", "she", "his", "her", "you", "your", "we", "our",
    "not", "no", "used", "use", "using", "can", "could", "should", "would", "any",
}


@dataclass
class RetrievedDoc:
    """一条检索结果：稳定 id（= 语料下标）+ 文本 + 分数 + 来源。"""

    id: int
    text: str
    score: float = 0.0
    source: str = SEMANTIC
    parent: Optional[int] = None


def _tokens(text: Any) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _dedup(seq: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _fts_query(terms: Iterable[str], mode: str = "or") -> str:
    """把若干实体/关键词拼成安全的 FTS5 MATCH 表达式。

    - mode="or" ：拆成单词后按 OR 连接（任一出现即可）——用于宽松召回/自动关键词
    - mode="and"：每个 term 作为**短语**（多词视为相邻词组），短语之间隐式 AND
                  ——对应论文 exact_query_keywords 的「文本中必须出现」语义
    """
    if mode == "and":
        phrases: List[str] = []
        for term in terms:
            toks = _tokens(term)
            if toks:
                phrases.append('"' + " ".join(toks) + '"')
        return " ".join(_dedup(phrases))
    toks = []
    for term in terms:
        toks.extend(_tokens(term))
    return " OR ".join(f'"{t}"' for t in _dedup(toks))


def split_sentences(paragraph: str) -> List[str]:
    """段内切句（Entity-Match 的粒度）。"""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(paragraph).strip())]
    return [p for p in parts if p]


class FTSIndex:
    """SQLite FTS5 词法索引：段落级（exact search）+ 句子级（entity match）。"""

    def __init__(self, docs: Sequence[str], sentences: Sequence[Tuple[int, str]]):
        self._docs = list(docs)
        self._sentences = list(sentences)
        self._con: Optional[sqlite3.Connection] = None
        # LangGraph 会在线程池里执行同步节点，且评测用 asyncio 并发多题：
        # 连接需跨线程共享 + 加锁串行化访问（语料小，开销可忽略）。
        self._lock = threading.Lock()

    def build(self) -> "FTSIndex":
        con = sqlite3.connect(":memory:", check_same_thread=False)
        con.execute("CREATE VIRTUAL TABLE doc_fts USING fts5(text)")
        con.executemany(
            "INSERT INTO doc_fts(rowid, text) VALUES (?, ?)",
            [(i + 1, d) for i, d in enumerate(self._docs)],
        )
        con.execute("CREATE VIRTUAL TABLE sent_fts USING fts5(text, docid UNINDEXED)")
        con.executemany(
            "INSERT INTO sent_fts(text, docid) VALUES (?, ?)",
            [(s, did) for did, s in self._sentences],
        )
        con.commit()
        self._con = con
        return self

    # bm25() 越小（越负）越相关，调用方需取负后使用
    def _run_doc(self, query: str, n: int) -> List[Tuple[int, float]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT rowid, bm25(doc_fts) FROM doc_fts "
                "WHERE doc_fts MATCH ? ORDER BY bm25(doc_fts) LIMIT ?",
                (query, int(n)),
            ).fetchall()
        return [(int(r[0]) - 1, float(r[1])) for r in rows]

    def _run_sent(self, query: str, n: int) -> List[Tuple[int, str]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT docid, text FROM sent_fts "
                "WHERE sent_fts MATCH ? ORDER BY bm25(sent_fts) LIMIT ?",
                (query, int(n)),
            ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def search(self, terms: Iterable[str], n: int, mode: str = "or") -> List[Tuple[int, float]]:
        if self._con is None:
            self.build()
        if not self._docs:
            return []
        query = _fts_query(terms, mode)
        rows = self._run_doc(query, n) if query else []
        # 「必须出现」过严时零结果 -> 回退到 OR，避免空集
        if not rows and mode == "and":
            fallback = _fts_query(terms, "or")
            rows = self._run_doc(fallback, n) if fallback else []
        return rows

    def _anchor_expr(self, entity: str, rank_terms: Sequence[str]) -> Tuple[str, bool]:
        """返回 (MATCH 表达式, 是否带排序词)。必须含 entity 短语；有额外事实词时 OR 加权。"""
        phrase = _fts_query([entity], "and")
        if not phrase:
            return "", False
        entity_tokens = set(_tokens(entity))
        extra = [t for t in (rank_terms or []) if set(_tokens(t)) - entity_tokens]
        if extra:
            return f'{phrase} AND ({_fts_query(extra, "or")})', True
        return phrase, False

    def docs_anchored(
        self, entity: str, rank_terms: Sequence[str], n: int
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """文档级锚定：返回 (命中行, 是否「有区分度」)。

        有区分度 = 表达式里带上了 rank_terms（如 born）且命中非空；否则说明退化成
        「只含实体短语」的宽匹配（几乎所有相关文档都命中），此时不应作为锚定依据。
        """
        if self._con is None:
            self.build()
        if not self._docs:
            return [], False
        expr, has_extra = self._anchor_expr(entity, rank_terms)
        if not expr:
            return [], False
        if has_extra:
            rows = self._run_doc(expr, n)
            if rows:
                return rows, True
        phrase, _ = self._anchor_expr(entity, [])
        return self._run_doc(phrase, n), False

    def sentences_anchored(
        self, entity: str, rank_terms: Sequence[str], n: int
    ) -> Tuple[List[Tuple[int, str]], bool]:
        """Anchored Matching（片段级）：必须含 entity **短语**，并按 rank_terms（judge 的
        事实词，如 "born"）过滤/加权；命中即「有区分度」。

        这是论文 Figure 6 的关键：把「被同名实体/无关新闻淹没」的片段换成
        「实体 + 事实词」同时命中的片段（如 Jeff Garlin (born June 5, 1962)）。
        """
        if self._con is None:
            self.build()
        if not self._sentences:
            return [], False
        expr, has_extra = self._anchor_expr(entity, rank_terms)
        if not expr:
            return [], False
        if has_extra:
            rows = self._run_sent(expr, n)
            if rows:
                return rows, True
        phrase, _ = self._anchor_expr(entity, [])
        return self._run_sent(phrase, n), False


class CorpusInteractionEngine:
    """论文 §3.1 的 Corpus Interaction Engine：把动作原语编译成一次检索。"""

    def __init__(
        self,
        docs: Sequence[str],
        semantic: VectorRetriever,
        *,
        fusion_pool: int = 20,
        default_weights: Tuple[float, float] = (0.5, 0.5),
        entity_top_k: int = 3,
        anchor_max: int = 2,
        anchor_bonus: float = 0.15,
        anchor_mode: str = "promote",
        snippet_mode: str = "off",
        memory: Any = None,
    ):
        self.docs = list(docs)
        self._semantic = semantic
        # GAM-RAG 风格经验库（ExperienceStore）；None = 关闭自进化（行为与旧版一致）
        self.memory = memory
        self.fusion_pool = int(fusion_pool)
        self.default_weights = (float(default_weights[0]), float(default_weights[1]))
        self.entity_top_k = int(entity_top_k)
        # 文档级锚定：promote=命中即前置（默认，消融里命中率最好）；
        #            bonus=只加 anchor_bonus 小加成；off=关闭
        self.anchor_max = int(anchor_max)
        self.anchor_bonus = float(anchor_bonus)
        self.anchor_mode = str(anchor_mode)
        # 片段级（句子）注入：off=关闭（默认，消融显示句子碎片会伤害 AG）；prepend=置于结果最前
        self.snippet_mode = str(snippet_mode)
        self._fts = FTSIndex(self.docs, self._build_sentences())

    # ------------------------------------------------------------------ 构建
    def _build_sentences(self) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        for idx, doc in enumerate(self.docs):
            for sent in split_sentences(doc):
                out.append((idx, sent))
        return out

    def build(self) -> "CorpusInteractionEngine":
        if self._semantic.matrix is None:
            self._semantic.build()
        self._fts.build()
        return self

    # ------------------------------------------------------------------ 检索
    def search(self, plan: Dict[str, Any]) -> List[RetrievedDoc]:
        """论文式入口：按 plan 里声明的动作原语执行一次「多面检索 + 上下文整形」。

        plan 支持字段::

            semantic_query : str  or None   稠密检索 query
            keywords_must  : List[str]      **必须出现**的 term（短语/隐式 AND），高权重
            keywords_any   : List[str]      替代说法（OR），权重打对折
            keywords       : List[str]      兼容旧字段，配合 keywords_mode
            keywords_mode  : "and"|"or"     旧字段语义：and=必须出现，or=任一出现
            weights        : (w_sem, w_exact) 融合权重，缺省 default_weights
            entity         : List[str]      Anchored Matching 实体词
            include        : List[int]      强制保留的 doc id
            exclude        : List[int]      过滤掉的 doc id
            top_k          : int            返回条数（adjust scale）
        """
        top_k = int(plan.get("top_k") or 5)
        pool = int(plan.get("pool") or self.fusion_pool)
        excluded = {int(i) for i in (plan.get("exclude") or [])}
        included = [int(i) for i in (plan.get("include") or []) if 0 <= int(i) < len(self.docs)]
        weights = plan.get("weights") or self.default_weights
        sem_q = plan.get("semantic_query") or ""

        # 词法分支拆成两类（对齐论文 exact_query_keywords 的语义）：
        #   must = 必须出现（每个 term 作短语，隐式 AND）—— 如事实词 "born"
        #   any  = 替代说法（OR，权重打对折）—— 如 judge 给的逗号分隔同义词组
        must = [str(t) for t in (plan.get("keywords_must") or [])]
        any_terms = [str(t) for t in (plan.get("keywords_any") or [])]
        legacy = plan.get("keywords")
        if legacy is not None:
            bucket = must if str(plan.get("keywords_mode") or "and") == "and" else any_terms
            bucket.extend(str(t) for t in legacy)
        if not must and not any_terms and plan.get("auto_keywords", True) and sem_q:
            any_terms = [t for t in _tokens(sem_q) if t not in _STOPWORDS]

        # --- 1) Multi-Faceted Retrieval：语义 + must(AND) + any(OR)
        sem_scores: Dict[int, float] = {}
        if sem_q:
            for idx, score in self._semantic.query_with_ids(sem_q, pool):
                sem_scores[idx] = float(score)

        def _bm25_norm(terms: List[str], mode: str) -> Dict[int, float]:
            rows = self._fts.search(terms, pool, mode=mode) if terms else []
            raw = {idx: -float(bm) for idx, bm in rows}  # bm25 越小越相关，取负
            return dict(zip(list(raw), _minmax(list(raw.values()))))

        exact_scores: Dict[int, float] = {}
        for idx, value in _bm25_norm(must, "and").items():
            exact_scores[idx] = max(exact_scores.get(idx, 0.0), value)
        for idx, value in _bm25_norm(any_terms, "or").items():
            exact_scores[idx] = max(exact_scores.get(idx, 0.0), 0.5 * value)

        # --- 2) weighted fusion：各自 min-max 后加权和
        fused: Dict[int, float] = {}
        if sem_scores:
            ids = list(sem_scores)
            for idx, value in zip(ids, _minmax([sem_scores[i] for i in ids])):
                fused[idx] = fused.get(idx, 0.0) + float(weights[0]) * value
        if exact_scores:
            ids = list(exact_scores)
            for idx, value in zip(ids, _minmax([exact_scores[i] for i in ids])):
                fused[idx] = fused.get(idx, 0.0) + float(weights[1]) * value

        # --- 2.5) 记忆先验（GAM-RAG）：门控重排 / 持久排除 / 记忆激活注入
        #   g_i = 1 + (1-π_i)·cos(m_i, q)：被历史反馈拉近的段落抬升、推离的压低；
        #   空库时 gates/excludes/injects 全空，此块为 no-op（冷启动逐比特回归）。
        injected: List[int] = []
        if self.memory is not None and sem_q:
            priors = self.memory.retrieval_priors(self._semantic.embed_query(sem_q))
            excluded |= priors.excludes
            for idx in list(fused):
                gate = priors.gates.get(idx)
                if gate is not None:
                    fused[idx] *= gate
            # 记忆激活：历史验证过的高置信段落直接拉到「最高融合分 × cos」的高度。
            # 门控是乘性的，救不了语义分被 min-max 压到近零的段落，激活才是兜底。
            ceiling = max(fused.values()) if fused else 1.0
            for doc_id, cos in priors.injects:
                if doc_id in excluded or not (0 <= doc_id < len(self.docs)):
                    continue
                activated = ceiling * float(cos)
                if activated > fused.get(doc_id, float("-inf")):
                    fused[doc_id] = activated
                    injected.append(doc_id)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        rank_terms = must + any_terms

        # --- 3) Context Shaping：exclude 过滤 / include 强制保留
        ordered = [(i, s) for i, s in ordered if i not in excluded]

        # --- 4) Anchored Matching（文档级）：
        #     anchor_mode="promote" 命中即提到最前（默认）
        #     anchor_mode="bonus"   只加 anchor_bonus 小加成，不硬性重排
        #     anchor_mode="off"     关闭锚定
        anchor_mode = str(plan.get("anchor_mode") or self.anchor_mode)
        snippet_mode = str(plan.get("snippet_mode") or self.snippet_mode)
        anchored_ids: List[int] = []
        if anchor_mode != "off":
            for entity in (plan.get("entity") or []):
                rows, discriminating = self._fts.docs_anchored(entity, rank_terms, pool)
                if not discriminating:
                    continue
                for idx, _bm in rows[: self.anchor_max]:
                    if idx not in excluded and idx not in anchored_ids:
                        anchored_ids.append(idx)
        if anchored_ids:
            if anchor_mode == "promote":
                score_map = dict(ordered)
                aset = set(anchored_ids)
                ordered = [(i, score_map.get(i, 1.0)) for i in anchored_ids] + [
                    (i, s) for i, s in ordered if i not in aset
                ]
            else:
                scores = dict(ordered)
                for idx in anchored_ids:
                    scores[idx] = scores.get(idx, 0.0) + self.anchor_bonus
                ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        present = {i for i, _ in ordered}
        for idx in included:
            if idx not in present and idx not in excluded:
                ordered.insert(0, (idx, 1.0))
                present.add(idx)
        ordered = ordered[: max(0, top_k)]

        results: List[RetrievedDoc] = []
        for idx, score in ordered:
            if idx in injected:
                source = MEMORY
            else:
                source = FUSION if (idx in sem_scores and idx in exact_scores) else (
                    SEMANTIC if idx in sem_scores else EXACT
                )
            results.append(RetrievedDoc(id=idx, text=self.docs[idx], score=float(score), source=source))

        # --- 5) Anchored Matching（片段级）：默认关闭（句子碎片会伤害 DS/AG）；
        #     仅当 snippet_mode="prepend" 时把实体短语（+事实词）命中的短句置于最前
        seen = {d.text for d in results}
        anchors: List[RetrievedDoc] = []
        budget = self.entity_top_k if (snippet_mode == "prepend" and anchor_mode != "off") else 0
        for entity in (plan.get("entity") or []):
            if budget <= 0:
                break
            rows, discriminating = self._fts.sentences_anchored(entity, rank_terms, budget)
            if not discriminating:
                # 无区分度（退化成「只含实体短语」的宽匹配）时不注入片段，避免噪声
                continue
            for parent, sentence in rows:
                if parent in excluded or sentence in seen:
                    continue
                seen.add(sentence)
                anchors.append(
                    RetrievedDoc(id=parent, text=sentence, score=0.0, source=ENTITY, parent=parent)
                )
                budget -= 1
        return anchors + results

    # ------------------------------------------------------------------ 垫片
    def query(self, question: str, n: int = 5) -> List[str]:
        """兼容既有 R 模块 / DS / AG：只走稠密检索，返回纯文本。"""
        return [d.text for d in self.search({"semantic_query": question, "top_k": n})]

    @property
    def size(self) -> int:
        return len(self.docs)

    @property
    def semantic(self) -> VectorRetriever:
        """底层向量检索器（Runtime 的记忆反馈要用它的 embedding 矩阵）。"""
        return self._semantic


def build_engine_from_config(cfg) -> CorpusInteractionEngine:
    """按配置（RAG_CORPUS + embeddings 服务）构建引擎；CLI --retrieval-mode engine 用。"""
    if not cfg.corpus_path:
        raise RuntimeError(
            "retrieval_mode=engine 需要提供语料：设置 RAG_CORPUS=<jsonl 路径>，"
            "或改用 --retrieval-mode vector/http/mock/none。"
        )
    semantic = VectorRetriever.from_jsonl(
        cfg.corpus_path,
        base_url=cfg.embed_base_url,
        model=cfg.embed_model,
        api_key=cfg.embed_api_key,
        batch_size=cfg.embed_batch_size,
    )
    return CorpusInteractionEngine(list(semantic.docs), semantic).build()
