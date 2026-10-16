"""LangGraph Agentic RAG：自进化经验库（对齐 GAM-RAG 的 Gain-Adaptive Memory）。

把 GAM-RAG 的句子级任务记忆降格到我们的检索单元（段落），training-free：

1) 记忆状态（论文 Eq.2，去掉时间通道）：每段维护 (m_i, π_i)
   - m_i 初始化为段落 embedding（L2 归一化）
   - π_i ∈ [0,1] 为不确定性（perplexity），初始 1
2) 检索先验（Eq.4 的门控）：g_i = 1 + (1-π_i)·cos(m_i, q)
   - 被历史成功 query 拉近的段落门控抬升（attraction）
   - 被负反馈推离的段落门控压低；cos ≤ -θ 且 π 足够低时进入「持久排除」
     （等价于 INTERACT-RAG exclude_docs 的跨查询进化版）
   - π 低且 cos 高的段落直接注入候选池（memory activation，补足语义池漏召回）
3) Kalman 增益更新（Eq.10-13）：
       e = y - cos(q, m);  K = π/(π+R)
       m ← norm(m + K·e·q);  π ← clip((1-K)·π + Q, 0, 1)
   非对称观测噪声 R_neg > R_pos：负反馈更保守，避免误杀潜在桥接证据；
   过程噪声 Q 保证 π 不塌缩到 0（记忆永远保留一点可塑性）。
4) 冷启动安全：无记忆段落不参与门控（等价 g≡1）；空库时检索行为与无记忆版
   逐比特一致，不破坏既有三档 A/B 的可比性。
5) 策略记忆（P1）：semantic_key → judge plan 的同构 Kalman 条目。子问题级
   检索成功后，把生效的 keywords/entity_boost 记入策略库；语义相似的新子问题
   首轮检索直接 warm-start（跳过「judge 不足→LLM 重写→再检索」的回合）。
   仅 π 收敛（≥2 次成功反馈）且 cos 达标的策略可被信任，失败策略会被否决。

持久化：可选 SQLite（doc_memory 表，跨进程进化）；缺省纯进程内存（多轮评测够用）。
线程安全：LangGraph 在线程池执行同步节点 + 评测 asyncio 并发，单锁保护全部读写。

离线自测（无需任何服务）::

    python -m langgraph_rag.memory
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class MemoryPriors:
    """一次检索的记忆先验：门控系数 / 持久排除 / 记忆激活注入。"""

    gates: Dict[int, float] = field(default_factory=dict)
    excludes: Set[int] = field(default_factory=set)
    injects: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class _DocMemory:
    vec: np.ndarray  # L2 归一化的任务记忆 m_i
    pi: float        # 不确定性 π_i
    pos: int = 0     # 累计正反馈次数（审计用）
    neg: int = 0     # 累计负反馈次数


@dataclass
class _StrategyMemory:
    """策略记忆条目：semantic_key → 曾经验证成功的检索策略（judge plan）。

    与段落记忆的差别：段落记忆的 m 初始化为文档向量（与 q 有正交分量），
    沿 q 的 Kalman 修正改变 cos(m,q)，方向通道有意义；而策略的键就是查询
    本身，m 若沿 q 更新则 cos 恒为 1（退化）。因此策略记忆把状态降为
    标量支持度 s = E[y]（该策略的期望收益），跑同构的 1D Kalman：
        e = y - s;  K = π/(π+R);  s ← s + K·e;  π ← clip((1-K)π + Q, 0, 1)
    vec 只作为语义匹配键（不更新），π/s 决定该策略是否可信。
    """

    key: str          # 情节的语义键（子问题 semantic_query 原文，审计/去重用）
    vec: np.ndarray   # 键的 embedding（仅匹配用，不随反馈更新）
    pi: float         # 不确定性 π
    s: float = 0.0    # 标量支持度：该策略的期望收益估计
    plan: Dict[str, Any] = field(default_factory=dict)
    pos: int = 0
    neg: int = 0


class ExperienceStore:
    """段落级 Gain-Adaptive 记忆库。

    参数（默认值对齐论文 Appendix C：R_pos=0.5, R_neg=1.0）::

        r_pos / r_neg  : 观测噪声（非对称，负反馈更保守）
        q_noise        : 过程噪声 Q，π 的地板（防止记忆僵化）
        exclude_cos    : 持久排除的 cos(m,q) 阈值（负得越多越该排）
        exclude_pi     : 持久排除要求的最大 π（只信"确定被推离"的记忆）
        inject_max     : 每次检索最多注入几个记忆激活段落
        inject_cos     : 注入要求的最小 cos(m,q)
        inject_pi      : 注入要求的最大 π

    策略记忆（P1）：semantic_key → judge plan 的标量 Kalman 条目::

        strategy_match_cos   : 反馈更新时认定「同一子问题」的 cos 阈值
        strategy_min_cos     : warm-start 要求的最小语义相关性 cos(key,q)
        strategy_max_pi      : warm-start 要求的最大 π（首次成功 π=0.383，
                               第二次成功 π=0.267 才达标——只 warm-start
                               被至少验证过两次的策略）
        strategy_min_support : warm-start 要求的最小支持度 s（负反馈把 s
                               拉回阈值下即刻否决）
        strategy_min_pos     : P3.1 warm-start 要求的最小独立成功次数 pos
                               （默认 3。消融证明 musique 上 2 次 agent 自我
                               标注成功不足以保 plan 质量——ws 污染使 hitR
                               1.00→0.97、JudgeAcc −7.5pp；3 次门槛掐断低质
                               策略滚雪球，gold/高质量反馈下不影响 ws 收益）
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        r_pos: float = 0.5,
        r_neg: float = 1.0,
        q_noise: float = 0.05,
        exclude_cos: float = -0.25,
        exclude_pi: float = 0.35,
        inject_max: int = 2,
        inject_cos: float = 0.65,
        inject_pi: float = 0.5,
        strategy_match_cos: float = 0.8,
        strategy_min_cos: float = 0.65,
        strategy_max_pi: float = 0.35,
        strategy_min_support: float = 0.5,
        strategy_min_pos: int = 3,
        strategy_max_entries: int = 64,
    ):
        self.db_path = str(db_path) if db_path else None
        self.r_pos = float(r_pos)
        self.r_neg = float(r_neg)
        self.q_noise = float(q_noise)
        self.exclude_cos = float(exclude_cos)
        self.exclude_pi = float(exclude_pi)
        self.inject_max = int(inject_max)
        self.inject_cos = float(inject_cos)
        self.inject_pi = float(inject_pi)
        self.strategy_match_cos = float(strategy_match_cos)
        self.strategy_min_cos = float(strategy_min_cos)
        self.strategy_max_pi = float(strategy_max_pi)
        self.strategy_min_support = float(strategy_min_support)
        self.strategy_min_pos = int(strategy_min_pos)
        # P2.1b：策略库容量上限（musique agent 臂曾膨胀到 123 条并伴随污染）；
        # 超限时按 (支持度 s 低, π 高) 优先淘汰
        self.strategy_max_entries = int(strategy_max_entries)

        self._mem: Dict[int, _DocMemory] = {}
        self._strategies: Dict[str, _StrategyMemory] = {}
        # P2.2：defer 模式（评测 turn 内隔离）——非 None 时写入只进暂存队列，
        # commit_defer() 统一应用；读路径（lookup/priors）只看已提交状态。
        self._deferred: Optional[List[tuple]] = None
        self._lock = threading.Lock()
        self._con: Optional[sqlite3.Connection] = None
        if self.db_path:
            self._con = sqlite3.connect(self.db_path, check_same_thread=False)
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS doc_memory ("
                " doc_id INTEGER PRIMARY KEY,"
                " vec BLOB NOT NULL,"
                " pi REAL NOT NULL,"
                " pos INTEGER NOT NULL DEFAULT 0,"
                " neg INTEGER NOT NULL DEFAULT 0,"
                " updated REAL)"
            )
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS strategies ("
                " key TEXT PRIMARY KEY,"
                " vec BLOB NOT NULL,"
                " pi REAL NOT NULL,"
                " s REAL NOT NULL DEFAULT 0,"
                " plan TEXT NOT NULL,"
                " pos INTEGER NOT NULL DEFAULT 0,"
                " neg INTEGER NOT NULL DEFAULT 0,"
                " updated REAL)"
            )
            self._con.commit()
            self._load()

    # ------------------------------------------------------------------ Kalman 更新
    def feedback(
        self, doc_id: int, doc_vec: np.ndarray, q_vec: np.ndarray, y: float
    ) -> None:
        """一次情节反馈（论文 §3.3.1）：y=+1 支持 / y=-1 不支持。

        m 沿 q 方向做 1D 投影修正（Eq.12），正交分量不动——对无关语义漂移鲁棒。
        """
        doc_id = int(doc_id)
        q = _unit(np.asarray(q_vec, dtype=np.float32))
        y = float(y)
        if self._deferred is not None:
            self._deferred.append(("doc", doc_id, np.asarray(doc_vec, dtype=np.float32), q, y))
            return
        with self._lock:
            mem = self._mem.get(doc_id)
            if mem is None:
                mem = _DocMemory(vec=_unit(np.asarray(doc_vec, dtype=np.float32)), pi=1.0)
                self._mem[doc_id] = mem
            s = float(np.dot(q, mem.vec))          # 当前投影支持度（Eq.15）
            e = y - s                              # 残差（Eq.10）
            r = self.r_pos if y > 0 else self.r_neg
            k = mem.pi / (mem.pi + r)              # Kalman 增益（Eq.11）
            mem.vec = _unit(mem.vec + (k * e) * q) # 状态更新（Eq.12）
            mem.pi = float(min(1.0, max(0.0, (1.0 - k) * mem.pi + self.q_noise)))  # Eq.13
            if y > 0:
                mem.pos += 1
            else:
                mem.neg += 1
            self._flush_one(doc_id, mem)

    # ------------------------------------------------------------------ 检索先验
    def retrieval_priors(self, q_vec: np.ndarray) -> MemoryPriors:
        """对当前 query 计算门控 / 持久排除 / 记忆激活（论文 Eq.4 的门控 g_i）。"""
        q = _unit(np.asarray(q_vec, dtype=np.float32))
        priors = MemoryPriors()
        with self._lock:
            for doc_id, mem in self._mem.items():
                cos = float(np.dot(q, mem.vec))
                priors.gates[doc_id] = 1.0 + (1.0 - mem.pi) * cos
                if cos <= self.exclude_cos and mem.pi <= self.exclude_pi:
                    priors.excludes.add(doc_id)
                elif (
                    cos >= self.inject_cos
                    and mem.pi <= self.inject_pi
                    and len(priors.injects) < self.inject_max
                ):
                    priors.injects.append((doc_id, cos))
            priors.injects.sort(key=lambda kv: -kv[1])
        return priors

    def cosine(self, doc_id: int, q_vec: np.ndarray) -> float:
        q = _unit(np.asarray(q_vec, dtype=np.float32))
        with self._lock:
            mem = self._mem.get(int(doc_id))
            return float(np.dot(q, mem.vec)) if mem else 0.0

    # ------------------------------------------------------------------ 策略记忆（P1）
    def lookup_strategy(self, q_vec: np.ndarray) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """为语义键 q 检索最可信的已验证策略。

        可信四重门：语义相关 cos(key,q) ≥ strategy_min_cos、π 已收敛
        （≤ strategy_max_pi）、支持度 s ≥ strategy_min_support、独立成功
        次数 pos ≥ strategy_min_pos（P3.1：防低质策略两次侥幸成功即污染 ws）。
        返回 (key, plan副本, cos) 或 None。
        """
        q = _unit(np.asarray(q_vec, dtype=np.float32))
        best: Optional[Tuple[float, str, Dict[str, Any]]] = None
        with self._lock:
            for key, sm in self._strategies.items():
                if (
                    sm.pi > self.strategy_max_pi
                    or sm.s < self.strategy_min_support
                    or sm.pos < self.strategy_min_pos
                ):
                    continue
                cos = float(np.dot(q, sm.vec))
                if cos >= self.strategy_min_cos and (best is None or cos > best[0]):
                    best = (cos, key, dict(sm.plan))
        return (best[1], best[2], best[0]) if best else None

    def update_strategy(
        self, key_text: str, q_vec: np.ndarray, plan: Dict[str, Any], y: float
    ) -> None:
        """策略情节反馈：y=+1 该 plan 带来成功检索 / y=-1 失败。

        标量支持度 s 跑与 Eq.10-13 同构的 1D Kalman（键向量只匹配不更新）；
        正反馈时把 plan 替换为最新成功版本。仅成功情节会新建条目——失败的
        策略不值得记，除非它曾经成功过（此时负反馈会压低 s 将其否决）。
        """
        q = _unit(np.asarray(q_vec, dtype=np.float32))
        plan = {k: v for k, v in (plan or {}).items() if v not in (None, "", [], {})}
        y = float(y)
        if self._deferred is not None:
            self._deferred.append(("strategy", key_text, q, dict(plan), y))
            return
        with self._lock:
            best_key, best_cos = None, -1.0
            for k, sm in self._strategies.items():
                cos = float(np.dot(q, sm.vec))
                if cos > best_cos:
                    best_key, best_cos = k, cos
            if best_key is not None and best_cos >= self.strategy_match_cos:
                sm = self._strategies[best_key]
                e = y - sm.s                               # 残差（Eq.10 同构）
                r = self.r_pos if y > 0 else self.r_neg
                k_gain = sm.pi / (sm.pi + r)                 # Kalman 增益（Eq.11）
                sm.s += k_gain * e                           # 支持度更新
                sm.pi = float(min(1.0, max(0.0, (1.0 - k_gain) * sm.pi + self.q_noise)))
                if y > 0:
                    sm.pos += 1
                    if plan:
                        sm.plan = dict(plan)
                else:
                    sm.neg += 1
                self._flush_strategy(sm)
                return
            if y > 0 and plan:
                # 首次成功：从先验 s=0, π=1 做第一次 Kalman 更新入库
                k_gain = 1.0 / (1.0 + self.r_pos)
                sm = _StrategyMemory(
                    key=key_text, vec=q,
                    pi=min(1.0, (1.0 - k_gain) * 1.0 + self.q_noise),
                    s=k_gain * 1.0,
                    plan=dict(plan), pos=1,
                )
                self._strategies[key_text] = sm
                self._flush_strategy(sm)
                self._evict_strategies_locked()

    def _evict_strategies_locked(self) -> None:
        """P2.1b：容量超限时淘汰最不可信条目（s 低者优先，同 s 淘汰 π 高者）。"""
        if self.strategy_max_entries <= 0:
            return
        while len(self._strategies) > self.strategy_max_entries:
            victim = min(self._strategies.items(), key=lambda kv: (kv[1].s, -kv[1].pi))[0]
            del self._strategies[victim]
            if self._con is not None:
                self._con.execute("DELETE FROM strategies WHERE key=?", (victim,))
                self._con.commit()

    # ------------------------------------------------------------------ defer/commit（P2.2 turn 内隔离）
    def begin_defer(self) -> None:
        """开始暂存：此后的 feedback/update_strategy 只入队不生效。

        评测多题并发共享记忆时，turn 内先完成的题会污染后跑的题
        （观察到 t1 就出现 warm-start）。turn 级隔离下，一轮内所有题
        看到的是同一份「轮初已提交」记忆，轮末统一提交。
        """
        with self._lock:
            if self._deferred is None:
                self._deferred = []

    def commit_defer(self) -> int:
        """应用全部暂存写入并退出 defer 模式，返回应用的写入数。"""
        with self._lock:
            pending, self._deferred = self._deferred, None
        if not pending:
            return 0
        for op in pending:
            if op[0] == "doc":
                _, doc_id, doc_vec, q, y = op
                self.feedback(doc_id, doc_vec, q, y)
            else:
                _, key_text, q, plan, y = op
                self.update_strategy(key_text, q, plan, y)
        return len(pending)

    @property
    def deferring(self) -> bool:
        return self._deferred is not None

    def _flush_strategy(self, sm: _StrategyMemory) -> None:
        if self._con is None:
            return
        self._con.execute(
            "INSERT INTO strategies(key, vec, pi, s, plan, pos, neg, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            " vec=excluded.vec, pi=excluded.pi, s=excluded.s, plan=excluded.plan,"
            " pos=excluded.pos, neg=excluded.neg, updated=excluded.updated",
            (
                sm.key,
                sm.vec.astype(np.float32).tobytes(),
                sm.pi,
                sm.s,
                json.dumps(sm.plan, ensure_ascii=False),
                sm.pos,
                sm.neg,
                time.time(),
            ),
        )
        self._con.commit()

    # ------------------------------------------------------------------ 统计/重置
    def stats(self) -> Dict[str, float]:
        with self._lock:
            out = {
                "n_docs": len(self._mem),
                "mean_pi": float(np.mean([m.pi for m in self._mem.values()])) if self._mem else 1.0,
                "pos": float(sum(m.pos for m in self._mem.values())),
                "neg": float(sum(m.neg for m in self._mem.values())),
                "n_strategies": len(self._strategies),
                "strategy_pos": float(sum(s.pos for s in self._strategies.values())),
                "strategy_neg": float(sum(s.neg for s in self._strategies.values())),
            }
            return out

    def reset(self) -> None:
        with self._lock:
            self._mem.clear()
            self._strategies.clear()
            if self._con is not None:
                self._con.execute("DELETE FROM doc_memory")
                self._con.execute("DELETE FROM strategies")
                self._con.commit()

    # ------------------------------------------------------------------ 持久化
    def _flush_one(self, doc_id: int, mem: _DocMemory) -> None:
        if self._con is None:
            return
        self._con.execute(
            "INSERT INTO doc_memory(doc_id, vec, pi, pos, neg, updated) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            " vec=excluded.vec, pi=excluded.pi, pos=excluded.pos,"
            " neg=excluded.neg, updated=excluded.updated",
            (
                doc_id,
                mem.vec.astype(np.float32).tobytes(),
                mem.pi,
                mem.pos,
                mem.neg,
                time.time(),
            ),
        )
        self._con.commit()

    def _load(self) -> None:
        assert self._con is not None
        rows = self._con.execute(
            "SELECT doc_id, vec, pi, pos, neg FROM doc_memory"
        ).fetchall()
        for doc_id, blob, pi, pos, neg in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            self._mem[int(doc_id)] = _DocMemory(vec=vec, pi=float(pi), pos=int(pos), neg=int(neg))
        srows = self._con.execute(
            "SELECT key, vec, pi, s, plan, pos, neg FROM strategies"
        ).fetchall()
        for key, blob, pi, s, plan, pos, neg in srows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            self._strategies[str(key)] = _StrategyMemory(
                key=str(key), vec=vec, pi=float(pi), s=float(s),
                plan=json.loads(plan), pos=int(pos), neg=int(neg),
            )
        if rows or srows:
            logger.info(
                "记忆库载入 %d 条段落记忆 + %d 条策略记忆（%s）",
                len(rows), len(srows), self.db_path,
            )


# ---------------------------------------------------------------------------- 自测
def _selftest() -> None:  # noqa: C901
    """离线自测：Kalman 动力学 / 门控 / 排除 / 注入 / 持久化 / 引擎集成。"""
    import hashlib
    import os
    import tempfile

    rng = np.random.default_rng(7)
    dim = 16
    q = _unit(rng.normal(size=dim).astype(np.float32))
    doc = _unit(rng.normal(size=dim).astype(np.float32))

    # 1) 吸引：连续正反馈使 cos(m,q) 单调上升、π 单调下降（Eq.16 的收敛性）
    store = ExperienceStore()
    cosines: List[float] = []
    for _ in range(5):
        store.feedback(0, doc, q, +1)
        cosines.append(store.cosine(0, q))
    assert all(b > a for a, b in zip(cosines, cosines[1:])), cosines
    # π 随更新下降 → K 变小 → 收敛减速（设计使然的保守性），5 次正反馈到 ~0.79
    assert cosines[-1] > 0.75, cosines
    priors = store.retrieval_priors(q)
    assert priors.gates[0] > 1.0
    assert any(doc_id == 0 for doc_id, _ in priors.injects), priors

    # 2) 推离：连续负反馈使 cos 单调下降；且比正反馈更慢（R_neg > R_pos）
    store2 = ExperienceStore()
    neg_cos: List[float] = []
    for _ in range(5):
        store2.feedback(1, doc, q, -1)
        neg_cos.append(store2.cosine(1, q))
    assert all(b < a for a, b in zip(neg_cos, neg_cos[1:])), neg_cos
    assert -neg_cos[1] < cosines[1], "负反馈应比正反馈保守"
    # π 保护：单次负反馈不足以进入持久排除
    assert 1 not in ExperienceStore(exclude_pi=0.35).retrieval_priors(q).excludes
    one = ExperienceStore()
    one.feedback(1, doc, q, -1)
    assert 1 not in one.retrieval_priors(q).excludes, "1 次负反馈不应持久排除"
    assert 1 in store2.retrieval_priors(q).excludes, "5 次负反馈应持久排除"

    # 3) 策略记忆：P3.1 三次独立成功才可 warm-start；正交查询不命中；失败后被否决
    st = ExperienceStore()
    plan = {"keywords": ["bridge", "entity"], "entity_boost": 0.2}
    st.update_strategy("sub question one", q, plan, +1)
    assert st.lookup_strategy(q) is None, "仅 1 次成功（π 未收敛）不可 warm-start"
    st.update_strategy("sub question one", q, plan, +1)
    assert st.lookup_strategy(q) is None, "仅 2 次成功（pos<3）不可 warm-start"
    st.update_strategy("sub question one", q, plan, +1)
    hit = st.lookup_strategy(q)
    assert hit is not None and hit[1] == plan, hit
    q_orth = _unit(rng.normal(size=dim).astype(np.float32))
    assert st.lookup_strategy(q_orth) is None, "无关查询不应 warm-start"
    for _ in range(4):
        st.update_strategy("sub question one", q, plan, -1)
    assert st.lookup_strategy(q) is None, "连续失败的策略应被否决"
    assert st.stats()["n_strategies"] == 1

    # 3b) P2.2 defer/commit：暂存期间写入不可见，提交后生效
    st_d = ExperienceStore()
    st_d.begin_defer()
    st_d.feedback(7, doc, q, +1)
    st_d.update_strategy("deferred key", q, plan, +1)
    assert st_d.stats()["n_docs"] == 0 and st_d.stats()["n_strategies"] == 0
    assert st_d.cosine(7, q) == 0.0, "defer 期间读路径不应看到暂存写入"
    assert st_d.commit_defer() == 2
    assert st_d.stats()["n_docs"] == 1 and st_d.stats()["n_strategies"] == 1
    assert not st_d.deferring
    assert st_d.commit_defer() == 0, "非 defer 状态提交应为 no-op"

    # 3c) P2.1b 容量淘汰：超限按支持度最低者出局
    st_c = ExperienceStore(strategy_max_entries=3)
    cap_qs = []
    for i in range(5):
        qi = _unit(rng.normal(size=dim).astype(np.float32))
        st_c.update_strategy(f"cap key {i}", qi, {"keywords": [f"k{i}"]}, +1)
        cap_qs.append(qi)
    assert st_c.stats()["n_strategies"] == 3, "超容应触发淘汰"
    # 压低幸存条目的 s 后，下一次新写入应优先淘汰它
    st_c.update_strategy("cap key 4", cap_qs[4], {"keywords": ["k4"]}, -1)
    q_new = _unit(rng.normal(size=dim).astype(np.float32))
    st_c.update_strategy("cap key new", q_new, {"keywords": ["kn"]}, +1)
    assert st_c.stats()["n_strategies"] == 3
    assert "cap key 4" not in st_c._strategies, "负反馈压低 s 的条目应被优先淘汰"

    # 4) 持久化 round-trip（段落 + 策略）
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mem.db")
        s1 = ExperienceStore(path)
        s1.feedback(3, doc, q, +1)
        s1.feedback(3, doc, q, -1)
        for _ in range(3):  # P3.1：3 次成功才过 ws 门槛
            s1.update_strategy("persist key", q, plan, +1)
        s2 = ExperienceStore(path)
        assert abs(s2.cosine(3, q) - s1.cosine(3, q)) < 1e-6
        assert s2.stats()["n_docs"] == 1
        hit2 = s2.lookup_strategy(q)
        assert hit2 is not None and hit2[1] == plan, hit2

    # 5) 引擎集成：门控抬升 / 持久排除 / 记忆激活注入（stub embedding，无服务）
    from .engine import MEMORY, CorpusInteractionEngine
    from .vector import VectorRetriever

    class _StubSemantic(VectorRetriever):
        def _embed(self, texts, desc=""):  # 确定性伪 embedding（md5 播种）
            out = []
            for t in texts:
                seed = int.from_bytes(hashlib.md5(t.encode("utf-8")).digest()[:8], "big")
                v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
                out.append(v / np.linalg.norm(v))
            return np.asarray(out, dtype=np.float32)

    docs = [
        f"doc {i}: " + " ".join(f"w{i}_{j}" for j in range(12)) for i in range(8)
    ]
    sem = _StubSemantic(docs=docs, verbose=False)
    sem.matrix = sem._embed(docs)
    engine = CorpusInteractionEngine(docs, sem)
    mem = ExperienceStore(inject_pi=0.6, exclude_pi=0.6)
    engine.memory = mem

    qtext = "selftest query one"
    qv = sem.embed_query(qtext)
    base_ids = [d.id for d in engine.search({"semantic_query": qtext, "top_k": 3})]
    target, victim = base_ids[0], base_ids[1]
    outsider = next(i for i in range(len(docs)) if i not in base_ids)
    for _ in range(4):
        mem.feedback(target, sem.matrix[target], qv, +1)
        mem.feedback(victim, sem.matrix[victim], qv, -1)
        mem.feedback(outsider, sem.matrix[outsider], qv, +1)
    evolved = engine.search({"semantic_query": qtext, "top_k": 3})
    ids = [d.id for d in evolved]
    assert victim not in ids, f"被推离的段落应被持久排除: {ids}"
    assert target in ids, f"被拉进的段落应保留: {ids}"
    assert outsider in ids and any(d.source == MEMORY for d in evolved), \
        f"语义池外的正反馈段落应被记忆激活注入: {evolved}"

    # 6) 空库回归：无记忆时检索结果与基线一致
    engine2 = CorpusInteractionEngine(docs, sem)
    assert [d.id for d in engine2.search({"semantic_query": qtext, "top_k": 3})] == base_ids
    engine2.memory = ExperienceStore()
    assert [d.id for d in engine2.search({"semantic_query": qtext, "top_k": 3})] == base_ids

    print("memory 自测通过：吸引/推离单调、非对称保守、持久排除、注入、策略记忆、defer/淘汰、持久化、空库回归 ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _selftest()
