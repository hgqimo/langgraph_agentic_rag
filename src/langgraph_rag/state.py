"""LangGraph Agentic RAG：State 定义。

- ChainState：局部「模块链」子图的状态（对应 async_rollout_single 里
  `for module in module_chain:` 的那段循环）
- RagState：顶层图状态（对应单条 question 的整条 rollout）
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class SubQuery(TypedDict, total=False):
    """对应原仓库 context["sub-query_docs_a"] 中的一项。"""

    query: str
    docs: List[str]
    doc_ids: List[int]
    answer: str


class TraceStep(TypedDict, total=False):
    """一次 Agent/Tool 调用的审计记录（对应 single_batch_list 的一项）。"""

    agent: str
    messages: List[Dict[str, str]]
    output: str
    is_valid: bool
    penalty: float
    extra: Dict[str, Any]


class ChainState(TypedDict, total=False):
    """模块链子图：按 module_chain 依次执行 QR / R / DS / AG。

    检索循环（INTERACT-RAG）：R 之后可进入 judge，判不足则 loop_rewrite 后回到 R，
    直到 judge 判定充分或 retrieval_round 达到 cfg.max_retrieval_rounds。
    max_retrieval_rounds == 1 时退化为现行为（单次检索，不调用 judge）。
    """

    query: str
    # 原始问题（linear 路径传入）：AG 的答案格式约束按原题判定题型——
    # QR 重写后的 query 会丢失 yes/no、二选一等题型特征
    original_question: str
    docs: List[str]
    # 与 docs 位置平行对齐的语料 id（引擎模式下为稳定 id；judge 的 exclude 下标
    # 指向 docs 位置，须经此映射转成语料 id 后才能传给引擎/记忆库）
    doc_ids: List[int]
    answer: str
    module_chain: List[str]
    module_idx: int
    qr_context: Dict[str, Any]
    retrieval_count: int
    # ---- 检索循环相关 ----
    retrieval_round: int
    retrieval_query: str
    tried_queries: List[str]
    exclude_ids: List[int]
    judge_continue: bool
    judge_missing: str
    judge_why: str
    judge_plan: Dict[str, Any]
    trace: Annotated[List[TraceStep], operator.add]


class RagState(TypedDict, total=False):
    """顶层状态：一次完整的 agentic RAG rollout。"""

    question: str
    gold_answer: str
    # 仅供离线评测：金标证据段落，用于统计检索命中率（不参与图内计算）
    gold_evidence: List[str]
    workflow: str
    workflow_cate: str
    module_chain: List[str]
    sub_queries: List[SubQuery]
    sub_idx: int
    current_query: str
    current_chain: List[str]
    docs: List[str]
    doc_ids: List[int]
    answer: str
    turn_number: int
    retrieval_count: int
    final_reward: float
    final_metrics: Dict[str, Any]
    error: str
    trace: Annotated[List[TraceStep], operator.add]
