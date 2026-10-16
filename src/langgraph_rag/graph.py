"""LangGraph Agentic RAG：图拓扑。

拓扑与 verl/trainer/ppo/ray_trainer.py::async_rollout_single 一一对应：

    START → plan ─┬─(QDS/QDP)→ decompose → ┌ sub_prepare → sub_plan → sub_run ┐ → summarize
                  │                        └──────────(sub_idx 未走完则回环)───┘
                  └─(线性链) → linear → finalize → END

其中「模块链」QR/R/DS/AG 被抽成可复用子图（build_chain_subgraph），
线性路径与每个子问题路径都调用它，等价于原实现的
`for module in module_chain:` 循环。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from . import agents
from .config import RagConfig
from .reward import compute_final_reward, summarize_metrics
from .runtime import Runtime
from .state import ChainState, RagState, SubQuery, TraceStep

logger = logging.getLogger(__name__)

MODULE_TO_NODE = {"QR": "rewrite", "R": "retrieve", "DS": "select_docs", "AG": "answer"}


def _record(
    agent: str,
    messages: List[Dict[str, str]],
    output: str,
    is_valid: bool,
    cfg: RagConfig,
    **extra: Any,
) -> TraceStep:
    """构造一条审计记录：合法给 0 penalty，非法给 cfg.invalid_penalty。"""
    return {
        "agent": agent,
        "messages": messages,
        "output": output,
        "is_valid": bool(is_valid),
        "penalty": 0.0 if is_valid else cfg.invalid_penalty,
        "extra": extra,
    }


def sanitize_chain(chain: Any, cfg: RagConfig) -> List[str]:
    """把 workflow 字符串规范成可执行的模块列表（未知模块丢弃，保证以 AG 收尾）。"""
    modules = chain.split(",") if isinstance(chain, str) else list(chain or [])
    no_retrieval = cfg.resolved_retrieval_mode() == "none"
    cleaned: List[str] = []
    for module in modules:
        module = str(module).strip()
        if module not in MODULE_TO_NODE:
            continue
        if module == "DS" and not cfg.enable_doc_selection:
            continue
        if no_retrieval and module in ("R", "DS"):
            # 无检索能力时退化为参数化直答（等价于原实现的 <workflow>AG</workflow>）
            continue
        cleaned.append(module)
    if "AG" not in cleaned:
        cleaned.append("AG")
    return cleaned


def _merge_retrieved(
    old_texts: List[str], old_ids: List[int], new_texts: List[str], new_ids: List[int]
) -> "tuple[List[str], List[int]]":
    """合并两轮检索结果，保持 texts 与语料 ids 位置平行对齐（跨轮累积证据）。"""
    texts = list(old_texts)
    ids = list(old_ids)
    seen = set(old_texts)
    for text, doc_id in zip(new_texts, new_ids):
        if text not in seen:
            seen.add(text)
            texts.append(text)
            ids.append(int(doc_id))
    return texts, ids


def build_chain_subgraph(runtime: Runtime):
    """模块链子图：按 module_chain 顺序执行 QR / R / DS / AG。

    当 cfg.max_retrieval_rounds > 1 时，R 之后进入运行期检索循环：
        R → judge ─(充足)→ 下一个模块
                   └(不足)→ loop_rewrite → R ...
    循环次数上限即 cfg.max_retrieval_rounds（=1 时退化为现行为，不调用 judge）。
    """
    cfg = runtime.cfg
    loop_enabled = cfg.max_retrieval_rounds > 1

    def route(state: ChainState) -> str:
        chain = state.get("module_chain") or []
        idx = state.get("module_idx", 0)
        if idx >= len(chain):
            return END
        return MODULE_TO_NODE[chain[idx]]

    def route_after_retrieve(state: ChainState):
        return "judge" if loop_enabled else route(state)

    def route_after_judge(state: ChainState):
        return "loop_rewrite" if state.get("judge_continue") else route(state)

    def rewrite(state: ChainState) -> Dict[str, Any]:
        query = str(state.get("query", ""))
        context = state.get("qr_context") or {}
        messages = agents.build_query_rewrite_messages(query, context)
        output = runtime.complete("QueryRewriteAgent", messages)
        parsed = agents.parse_query_rewrite(output, query)
        return {
            "query": str(parsed["query_rewrited"]),
            "module_idx": state.get("module_idx", 0) + 1,
            "trace": [_record("QueryRewriteAgent", messages, output, bool(parsed["is_valid"]), cfg)],
        }

    def retrieve(state: ChainState) -> Dict[str, Any]:
        # 循环内 retrieval_query 会被重写；首次检索用链上经过 QR 的 query
        query = str(state.get("retrieval_query") or state.get("query", ""))
        round_no = int(state.get("retrieval_round", 0)) + 1
        plan: Dict[str, Any] = {
            "semantic_query": query,
            "top_k": cfg.top_k,
            "exclude": list(state.get("exclude_ids") or []),
        }
        guide = dict(state.get("judge_plan") or {})
        warm_start = False
        if not guide and round_no == 1:
            # P1 warm-start：首轮检索没有 judge 引导 → 查策略库里验证过的 plan，
            # 命中则省掉「judge 不足 → LLM 重写 → 再检索」的整个回合
            hit = runtime.lookup_strategy(query)
            if hit:
                guide = dict(hit[1])
                warm_start = True
        for key, value in guide.items():
            if key in ("keywords_must", "keywords_any", "keywords", "entity", "weights", "include"):
                plan[key] = value

        docs_new = runtime.retrieve_docs(plan)
        texts_new = [str(d.text) for d in docs_new]
        ids_new = [int(d.id) for d in docs_new]
        merged, merged_ids = _merge_retrieved(
            list(state.get("docs") or []), list(state.get("doc_ids") or []), texts_new, ids_new
        )
        step: TraceStep = {
            "agent": "RetrievalTool",
            "messages": [{"role": "user", "content": query}],
            "output": f"round {round_no}: retrieved {len(texts_new)} documents",
            "is_valid": bool(texts_new),
            "penalty": 0.0,
            "extra": {
                "docs": texts_new,
                "doc_ids": ids_new,
                "round": round_no,
                "plan": {k: v for k, v in plan.items() if k != "semantic_query"},
                "warm_start": warm_start,
            },
        }
        # 未启用循环时 R 是普通模块：执行完直接推进 module_idx
        return {
            "docs": merged,
            "doc_ids": merged_ids,
            "retrieval_round": round_no,
            "retrieval_count": state.get("retrieval_count", 0) + 1,
            "module_idx": state.get("module_idx", 0) + (0 if loop_enabled else 1),
            "judge_plan": {},
            "trace": [step],
        }

    def judge(state: ChainState) -> Dict[str, Any]:
        """判断累积文档是否足够；不足则给出下一轮检索引导（原语/排除/缺失信息）。"""
        query = str(state.get("query", ""))
        docs = list(state.get("docs") or [])
        round_no = int(state.get("retrieval_round", 1))

        messages = agents.build_evidence_check_messages(
            query, docs, round_no, cfg.max_retrieval_rounds
        )
        output = runtime.complete("EvidenceJudgeAgent", messages)
        parsed = agents.parse_evidence_check(output)

        # 达到最大轮数时强制收尾，避免无限循环
        sufficient = bool(parsed["sufficient"]) or round_no >= cfg.max_retrieval_rounds
        max_words = cfg.judge_missing_max_words
        missing_words = " ".join(str(parsed.get("missing", "")).split()[:max_words])
        why_words = " ".join(str(parsed.get("why", "")).split()[:max_words])

        # judge 看到的 "Document i" 是累积 docs 列表的下标，须经 doc_ids 映射成
        # 语料 id 后才能传给引擎的 exclude 原语与记忆库的负反馈
        doc_ids = list(state.get("doc_ids") or [])
        exclude_positions = [int(i) for i in (parsed.get("exclude") or [])]
        exclude_corpus = [doc_ids[p] for p in exclude_positions if 0 <= p < len(doc_ids)]

        updates: Dict[str, Any] = {
            "judge_continue": not sufficient,
            "judge_missing": missing_words,
            "judge_why": why_words,
            "judge_plan": dict(parsed.get("plan") or {}),
            "exclude_ids": sorted(
                set(int(i) for i in (state.get("exclude_ids") or [])) | set(exclude_corpus)
            ),
            "trace": [
                _record(
                    "EvidenceJudgeAgent",
                    messages,
                    output,
                    bool(parsed["is_valid"]),
                    cfg,
                    sufficient=sufficient,
                    round=round_no,
                    why=why_words,
                    exclude=exclude_corpus,
                )
            ],
        }
        if sufficient:
            updates["module_idx"] = state.get("module_idx", 0) + 1
        return updates

    def loop_rewrite(state: ChainState) -> Dict[str, Any]:
        """循环内重写检索 query（结合已试过的 query、还缺什么、以及失败诊断）。"""
        query = str(state.get("query", ""))
        tried = list(state.get("tried_queries") or [query])
        messages = agents.build_retry_query_messages(
            query, tried, state.get("judge_missing", ""), state.get("judge_why", "")
        )
        output = runtime.complete("QueryRewriteAgent", messages)
        parsed = agents.parse_query_rewrite(output, query)
        new_query = str(parsed["query_rewrited"])
        if new_query:
            tried.append(new_query)
        return {
            "retrieval_query": new_query,
            "tried_queries": tried,
            "trace": [
                _record("QueryRewriteAgent", messages, output, bool(parsed["is_valid"]), cfg, loop=True)
            ],
        }

    def select_docs(state: ChainState) -> Dict[str, Any]:
        docs = list(state.get("docs") or [])
        doc_ids = list(state.get("doc_ids") or [])
        query = str(state.get("query", ""))
        messages = agents.build_doc_selection_messages(query, docs)
        output = runtime.complete("DocumentSelectionAgent", messages)
        parsed = agents.parse_doc_ids(output, len(docs))
        ids = [int(i) for i in parsed["docs_ids"]]
        kept = [docs[i] for i in ids if 0 <= i < len(docs)]
        kept_ids = [doc_ids[i] for i in ids if 0 <= i < len(docs) and i < len(doc_ids)]
        return {
            "docs": kept,
            "doc_ids": kept_ids,
            "module_idx": state.get("module_idx", 0) + 1,
            "trace": [
                _record(
                    "DocumentSelectionAgent",
                    messages,
                    output,
                    bool(parsed["is_valid"]),
                    cfg,
                    selected=ids,
                )
            ],
        }

    def answer(state: ChainState) -> Dict[str, Any]:
        docs = list(state.get("docs") or [])
        # linear 路径：AG 回答**原始问题**——QR 重写是面向检索的（"Which film was
        # released more recently, A or B?" 会被改写成 "Release dates of A and B"），
        # 直接拿重写句当 Question 会把答案带偏成日期枚举；子问题路径无
        # original_question，仍按子问题回答（最终格式由 AS 按原题约束）。
        query = str(state.get("original_question") or state.get("query", ""))
        messages = agents.build_answer_messages(
            query, docs, format_query=str(state.get("original_question") or ""))
        output = runtime.complete("AnswerGenerationAgent", messages)
        parsed = agents.parse_answer(output)
        return {
            "answer": str(parsed["answer"]),
            "module_idx": state.get("module_idx", 0) + 1,
            "trace": [_record("AnswerGenerationAgent", messages, output, bool(parsed["is_valid"]), cfg)],
        }

    builder = StateGraph(ChainState)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("select_docs", select_docs)
    builder.add_node("answer", answer)

    path_map = {node: node for node in MODULE_TO_NODE.values()}
    path_map[END] = END
    if loop_enabled:
        builder.add_node("judge", judge)
        builder.add_node("loop_rewrite", loop_rewrite)
        path_map["judge"] = "judge"
        path_map["loop_rewrite"] = "loop_rewrite"

    builder.add_conditional_edges(START, route, path_map)
    builder.add_conditional_edges("retrieve", route_after_retrieve, path_map)
    if loop_enabled:
        builder.add_conditional_edges("judge", route_after_judge, path_map)
        builder.add_conditional_edges(
            "loop_rewrite", lambda _state: "retrieve", {"retrieve": "retrieve"}
        )
    for node in ("rewrite", "select_docs"):
        builder.add_conditional_edges(node, route, path_map)
    builder.add_conditional_edges("answer", route, path_map)
    return builder.compile()


def build_rag_graph(runtime: Runtime, checkpointer: Any = None):
    """组装主图并编译。"""
    cfg = runtime.cfg
    chain_graph = build_chain_subgraph(runtime)

    # ------------------------------------------------------------------ 规划
    def plan(state: RagState) -> Dict[str, Any]:
        question = state["question"]
        messages = agents.build_planning_messages(question, cfg.enable_qdp)
        output = runtime.complete("PlanningAgent", messages)
        parsed = agents.parse_workflow(output)
        is_valid = bool(parsed["is_valid"])
        workflow = str(parsed["workflow"]) if is_valid else cfg.default_workflow_on_invalid
        cate = workflow if workflow in ("QDP", "QDS") else "ELSE"
        return {
            "workflow": workflow,
            "workflow_cate": cate,
            "sub_idx": 0,
            "retrieval_count": 0,
            "trace": [_record("PlanningAgent", messages, output, is_valid, cfg, workflow=workflow)],
        }

    def route_after_plan(state: RagState) -> str:
        return "decompose" if state["workflow_cate"] in ("QDS", "QDP") else "linear"

    # --------------------------------------------------------------- 问题分解
    def decompose(state: RagState) -> Dict[str, Any]:
        question = state["question"]
        serial = state["workflow_cate"] == "QDS"
        agent_name = "QueryDecompositionAgentSerial" if serial else "QueryDecompositionAgentParallel"
        messages = agents.build_decomposition_messages(question, serial=serial)
        output = runtime.complete(agent_name, messages)
        parsed = agents.parse_subquestions(output, question, cfg.max_sub_questions)
        sub_queries: List[SubQuery] = [
            {"query": str(s), "docs": [], "answer": ""}
            for s in list(parsed["subquestions"])[: cfg.max_sub_questions]
        ]
        # 对齐原实现：QDS 为 1 + 子问题数，QDP 固定为 2
        turn_number = 1 + len(sub_queries) if serial else 2
        return {
            "sub_queries": sub_queries,
            "sub_idx": 0,
            "turn_number": turn_number,
            "trace": [
                _record(
                    agent_name,
                    messages,
                    output,
                    bool(parsed["is_valid"]),
                    cfg,
                    sub_queries=[s["query"] for s in sub_queries],
                )
            ],
        }

    # --------------------------------------------------- 子问题循环（QDS / QDP）
    def sub_prepare(state: RagState) -> Dict[str, Any]:
        subs = [dict(s) for s in state["sub_queries"]]
        idx = state["sub_idx"]
        item = subs[idx]
        query = str(item.get("query", ""))

        updates: Dict[str, Any] = {"current_query": query, "sub_queries": subs}

        # QDS 串行分解：先用 QR 结合已完成子问答做指代消解，再进入子工作流
        if state["workflow_cate"] == "QDS":
            context = {
                "original_query": state["question"],
                "type": state["workflow_cate"],
                "sub-query_docs_a": [dict(s) for s in subs],
            }
            messages = agents.build_query_rewrite_messages(query, context)
            output = runtime.complete("QueryRewriteAgent", messages)
            parsed = agents.parse_query_rewrite(output, query)
            query = str(parsed["query_rewrited"])
            item["query"] = query
            subs[idx] = item
            updates["current_query"] = query
            updates["sub_queries"] = subs
            updates["trace"] = [
                _record("QueryRewriteAgent", messages, output, bool(parsed["is_valid"]), cfg, index=idx)
            ]
        return updates

    def sub_plan(state: RagState) -> Dict[str, Any]:
        query = state["current_query"]
        messages = agents.build_sub_planning_messages(query)
        output = runtime.complete("SubPlanningAgent", messages)
        parsed = agents.parse_workflow(output)
        is_valid = bool(parsed["is_valid"])
        chain = sanitize_chain(parsed["workflow"], cfg)
        if cfg.sub_workflow_override:
            # 对齐原实现：SubPlanning 输出被硬编码覆盖，但 penalty 仍按解析结果记录
            chain = sanitize_chain(cfg.sub_workflow_override, cfg)
        return {
            "current_chain": chain,
            "trace": [_record("SubPlanningAgent", messages, output, is_valid, cfg, chain=chain)],
        }

    def sub_run(state: RagState) -> Dict[str, Any]:
        subs = [dict(s) for s in state["sub_queries"]]
        idx = state["sub_idx"]
        sub_state: ChainState = {
            "query": state["current_query"],
            "docs": [],
            "doc_ids": [],
            "answer": "",
            "module_chain": list(state["current_chain"]),
            "module_idx": 0,
            "qr_context": {},
            "retrieval_count": 0,
            "trace": [],
        }
        out = chain_graph.invoke(sub_state)

        item = subs[idx]
        item["query"] = state["current_query"]
        item["docs"] = list(out.get("docs") or [])
        item["doc_ids"] = list(out.get("doc_ids") or [])
        item["answer"] = str(out.get("answer", ""))
        subs[idx] = item
        return {
            "sub_queries": subs,
            "sub_idx": idx + 1,
            "retrieval_count": state.get("retrieval_count", 0) + int(out.get("retrieval_count", 0)),
            "trace": list(out.get("trace") or []),
        }

    def route_after_sub(state: RagState) -> str:
        return "sub_prepare" if state["sub_idx"] < len(state["sub_queries"]) else "summarize"

    # ------------------------------------------------------------- 答案汇总
    def summarize(state: RagState) -> Dict[str, Any]:
        subs = state["sub_queries"]
        messages = agents.build_answer_summarization_messages(
            state["question"],
            [str(s.get("query", "")) for s in subs],
            [str(s.get("answer", "")) for s in subs],
        )
        output = runtime.complete("AnswerSummarizationAgent", messages)
        parsed = agents.parse_answer(output)
        return {
            "answer": str(parsed["answer"]),
            "trace": [_record("AnswerSummarizationAgent", messages, output, bool(parsed["is_valid"]), cfg)],
        }

    # --------------------------------------------------------- 线性链（直答）
    def linear(state: RagState) -> Dict[str, Any]:
        chain = sanitize_chain(state["workflow"], cfg)
        sub_state: ChainState = {
            "query": state["question"],
            "original_question": state["question"],
            "docs": [],
            "doc_ids": [],
            "answer": "",
            "module_chain": chain,
            "module_idx": 0,
            "qr_context": {},
            "retrieval_count": 0,
            "trace": [],
        }
        out = chain_graph.invoke(sub_state)
        return {
            "module_chain": chain,
            "docs": list(out.get("docs") or []),
            "doc_ids": list(out.get("doc_ids") or []),
            "answer": str(out.get("answer", "")),
            "turn_number": 1,
            "retrieval_count": int(out.get("retrieval_count", 0)),
            "trace": list(out.get("trace") or []),
        }

    # ------------------------------------------------------------- 收尾打分
    def finalize(state: RagState) -> Dict[str, Any]:
        answer = str(state.get("answer", ""))
        gold = str(state.get("gold_answer", ""))
        final_reward = compute_final_reward(answer, gold)
        metrics = summarize_metrics(
            trace=list(state.get("trace") or []),
            cfg=cfg,
            final_reward=final_reward,
            turn_number=int(state.get("turn_number", 1)),
            retrieval_count=int(state.get("retrieval_count", 0)),
            workflow_cate=str(state.get("workflow_cate", "ELSE")),
        )
        # 自进化：把本轨迹的检索情节写回记忆库（memory_feedback=off 时为 no-op）
        try:
            updates = runtime.apply_memory_feedback(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆反馈写回失败（不影响本次回答）：%s", exc)
            updates = {}
        if updates:
            metrics = {**metrics, "memory_updates": updates}
        return {"final_reward": final_reward, "final_metrics": metrics}

    builder = StateGraph(RagState)
    builder.add_node("plan", plan)
    builder.add_node("decompose", decompose)
    builder.add_node("sub_prepare", sub_prepare)
    builder.add_node("sub_plan", sub_plan)
    builder.add_node("sub_run", sub_run)
    builder.add_node("summarize", summarize)
    builder.add_node("linear", linear)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", route_after_plan, {"decompose": "decompose", "linear": "linear"}
    )
    builder.add_edge("decompose", "sub_prepare")
    builder.add_edge("sub_prepare", "sub_plan")
    builder.add_edge("sub_plan", "sub_run")
    builder.add_conditional_edges(
        "sub_run", route_after_sub, {"sub_prepare": "sub_prepare", "summarize": "summarize"}
    )
    builder.add_edge("summarize", "finalize")
    builder.add_edge("linear", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
