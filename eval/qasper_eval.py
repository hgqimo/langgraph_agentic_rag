"""用 QASPER 数据集端到端评测 LangGraph Agentic RAG 工作流。

组成：
- 语料：QASPER 论文的段落（QASPER 的 evidence 就是段落原文，天然适合做检索金标）
- 检索：本机 BGE（OpenAI 兼容 /v1/embeddings）向量索引 + 余弦 top-k
- 生成：本机 vLLM（Qwen3.8-27B-FP8，关闭思考模式）
- 评测：答案 F1/EM（本仓库奖励口径）+ 证据命中率 + Agent 输出格式合法率 + turn/检索次数

用法::

    python eval/qasper_eval.py \\
        --base-url http://127.0.0.1:8000/v1 --model model/Qwen3.8-27B-FP8 \\
        --embed-url http://127.0.0.1:10200/v1 --embed-model text-embedding-3-small \\
        --papers 12 --questions 24 --workers 4 --scope open --out results/qasper_result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:      # 支持不经安装直接运行
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from langgraph_rag.config import AGENT_NAMES, RagConfig  # noqa: E402
from langgraph_rag.graph import build_rag_graph  # noqa: E402
from langgraph_rag.llm import build_llm  # noqa: E402
from langgraph_rag.reward import compute_scores  # noqa: E402
from langgraph_rag.runtime import Runtime  # noqa: E402
from langgraph_rag.vector import VectorRetriever  # noqa: E402

DEFAULT_DATA = str(_REPO_ROOT / "datasets" / "qasper" / "raw" / "qasper-dev-v0.3.json")
PLACEHOLDER_ONLY = re.compile(r"^\s*((BIBREF|TABREF|FIGREF|REF)\d+\s*,?\s*)+\s*$", re.I)


# ---------------------------------------------------------------------------- 数据
def load_papers(path: str, limit: int, seed: int) -> Dict[str, Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    keys = sorted(data)
    random.Random(seed).shuffle(keys)
    return {k: data[k] for k in keys[:limit]}


def pick_gold(answer: Dict[str, Any]) -> Tuple[str, str]:
    """返回 (gold, 类型)；无法评测的返回 ("", "")。"""
    if answer.get("unanswerable"):
        return "", ""
    if answer.get("yes_no") is not None:
        return ("yes" if answer["yes_no"] else "no"), "yesno"

    spans = [s.strip() for s in (answer.get("extractive_spans") or []) if s and s.strip()]
    spans = [s for s in spans if not PLACEHOLDER_ONLY.match(s)]
    if spans:
        return ", ".join(spans), "extractive"

    free_form = (answer.get("free_form_answer") or "").strip()
    if free_form and not PLACEHOLDER_ONLY.match(free_form):
        return free_form, "free_form"
    return "", ""


def build_corpus(papers: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """段落级切分；文本前缀论文标题/章节，便于向量检索区分来源。"""
    docs: List[str] = []
    meta: List[Dict[str, str]] = []
    for pid, paper in papers.items():
        title = paper.get("title", "")
        for section in paper.get("full_text", []):
            name = section.get("section_name", "")
            for para in section.get("paragraphs", []):
                text = (para or "").strip()
                if not text:
                    continue
                docs.append(f"[{title}] {name}: {text}")
                meta.append({"paper_id": pid, "title": title, "section": name, "raw": text})
    return docs, meta


def build_qas(
    papers: Dict[str, Dict[str, Any]], answer_type: str, max_words: int
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for pid, paper in papers.items():
        for qa in paper.get("qas", []):
            if not qa.get("answers"):
                continue
            gold, kind = pick_gold(qa["answers"][0]["answer"])
            if not gold:
                continue
            if answer_type == "yesno" and kind != "yesno":
                continue
            if answer_type == "short" and kind != "yesno" and len(gold.split()) > max_words:
                continue
            evidence = [
                e.strip()
                for e in (qa["answers"][0]["answer"].get("evidence") or [])
                if e and e.strip()
            ]
            items.append(
                {
                    "paper_id": pid,
                    "title": paper.get("title", ""),
                    "question": qa["question"],
                    "gold": gold,
                    "kind": kind,
                    "evidence": evidence,
                }
            )
    return items


# ---------------------------------------------------------------------------- 指标
def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def evidence_hits(state: Dict[str, Any], evidence: List[str]) -> Tuple[bool, bool]:
    """返回 (检索命中, 生成前命中)：金标证据是否出现在 R 的结果 / AG 的上下文里。"""
    if not evidence:
        return False, False
    golds = [_norm(e) for e in evidence]

    retrieved: List[str] = []
    ag_context: List[str] = []
    for step in state.get("trace") or []:
        if step.get("agent") == "RetrievalTool":
            retrieved.extend(_norm(d) for d in (step.get("extra") or {}).get("docs", []))
        elif step.get("agent") == "AnswerGenerationAgent":
            ag_context.append("\n".join(_norm(m.get("content", "")) for m in step.get("messages") or []))

    def hit(texts: List[str]) -> bool:
        return any(g in t for g in golds for t in texts)

    return hit(retrieved), hit(ag_context)


def evaluate_one(item: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    pred = str(state.get("answer", ""))
    metrics = state.get("final_metrics") or {}
    scores = compute_scores([pred], [item["gold"]])
    hit_r, hit_ag = evidence_hits(state, item["evidence"])

    steps = [s for s in (state.get("trace") or []) if s.get("agent") != "RetrievalTool"]
    valid_steps = [s for s in steps if s.get("is_valid")]
    invalid = [f"{s['agent']}:{str(s.get('output'))[:60]}" for s in steps if not s.get("is_valid")]

    return {
        "question": item["question"],
        "gold": item["gold"],
        "pred": pred,
        "kind": item["kind"],
        "workflow": state.get("workflow"),
        "workflow_cate": state.get("workflow_cate"),
        # final_f1 用 本仓库的奖励口径：yes/no 类取 EM，其余取 token-level F1
        "final_f1": float(metrics.get("final_f1", scores["f1"])),
        "f1_raw": scores["f1"],
        "em": scores["em"],
        "hit_retrieval": hit_r,
        "hit_generator": hit_ag,
        "n_evidence": len(item["evidence"]),
        "turn_number": metrics.get("turn_number"),
        "retrieval_times": metrics.get("retrieval_times"),
        "n_steps": len(steps),
        "n_invalid": len(invalid),
        "invalid_examples": invalid[:3],
        "total_penalty": sum(v for vals in (metrics.get("penalty") or {}).values() for v in vals),
        "warm_starts": sum(
            1 for s in state.get("trace") or []
            if s.get("agent") == "RetrievalTool" and (s.get("extra") or {}).get("warm_start")
        ),
    }


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    hit_pool = [r for r in rows if r["n_evidence"] > 0]
    total_steps = sum(r["n_steps"] for r in rows)
    total_invalid = sum(r["n_invalid"] for r in rows)

    summary: Dict[str, Any] = {
        "n_questions": n,
        "answer_f1": _mean([r["final_f1"] for r in rows]),
        "answer_em": _mean([r["em"] for r in rows]),
        "generation_accuracy_f1_ge_0.5": _mean([1.0 if r["final_f1"] >= 0.5 else 0.0 for r in rows]),
        f"evidence_hit_at_retrieval": _mean([1.0 if r["hit_retrieval"] else 0.0 for r in hit_pool]),
        "evidence_hit_at_generator": _mean([1.0 if r["hit_generator"] else 0.0 for r in hit_pool]),
        "format_valid_rate": 1 - total_invalid / total_steps if total_steps else 0.0,
        "mean_steps_per_question": _mean([r["n_steps"] for r in rows]),
        "mean_turn_number": _mean([r["turn_number"] or 0 for r in rows]),
        "mean_retrieval_times": _mean([r["retrieval_times"] or 0 for r in rows]),
        "total_format_penalty": sum(r["total_penalty"] for r in rows),
        "warm_starts": sum(r.get("warm_starts", 0) for r in rows),
        "workflow_distribution": {
            cate: sum(1 for r in rows if r["workflow_cate"] == cate)
            for cate in sorted({r["workflow_cate"] for r in rows})
        },
    }
    for kind in sorted({r["kind"] for r in rows}):
        subset = [r for r in rows if r["kind"] == kind]
        summary[f"answer_f1/{kind}"] = _mean([r["final_f1"] for r in subset])
    return summary


# ---------------------------------------------------------------------------- 运行
def _make_retriever(args: argparse.Namespace, docs: List[str], kind: str, verbose: bool = False):
    """构造检索器：vector=纯向量；engine=Corpus Interaction Engine（向量+BM25+entity）。"""
    semantic = VectorRetriever(
        docs=docs,
        base_url=args.embed_url,
        model=args.embed_model,
        batch_size=args.embed_batch,
        verbose=verbose,
    )
    if kind == "engine":
        from langgraph_rag.engine import CorpusInteractionEngine

        return CorpusInteractionEngine(list(docs), semantic).build()
    return semantic.build()


def _tagged_db_path(db_path: str, tag: str) -> str:
    """给持久化记忆库文件名加档位/论文标签，避免三档 A/B 或逐论文语料间互相污染。"""
    if not db_path or not tag:
        return db_path
    path = Path(db_path)
    return str(path.with_name(f"{path.stem}.{tag}{path.suffix}"))


def _memory_stats(runtimes: Dict[str, Runtime]) -> Dict[str, Any]:
    """聚合所有 Runtime 的记忆库状态（逐轮进化曲线的观测点）。"""
    stores = [rt.memory for rt in runtimes.values() if getattr(rt, "memory", None) is not None]
    if not stores:
        return {}
    stats = [s.stats() for s in stores]
    total = sum(st["n_docs"] for st in stats) or 1
    return {
        "n_docs": sum(st["n_docs"] for st in stats),
        "mean_pi": sum(st["mean_pi"] * st["n_docs"] for st in stats) / total,
        "pos_updates": sum(st["pos"] for st in stats),
        "neg_updates": sum(st["neg"] for st in stats),
        "n_strategies": sum(st.get("n_strategies", 0) for st in stats),
        "strategy_pos": sum(st.get("strategy_pos", 0) for st in stats),
        "strategy_neg": sum(st.get("strategy_neg", 0) for st in stats),
    }


async def run_eval(
    args: argparse.Namespace,
    retriever_kind: Optional[str] = None,
    max_rounds: Optional[int] = None,
    memory_tag: str = "",
) -> Dict[str, Any]:
    retriever_kind = retriever_kind or args.retriever
    max_rounds = int(max_rounds if max_rounds is not None else args.max_rounds)
    memory_mode = str(getattr(args, "memory", "off") or "off")
    turns = max(1, int(getattr(args, "turns", 1)))

    papers = load_papers(args.data, args.papers, args.seed)
    docs, meta = build_corpus(papers)
    qas = build_qas(papers, args.answer_type, args.max_gold_words)
    random.Random(args.seed).shuffle(qas)
    qas = qas[: args.questions]
    print(
        f"论文 {len(papers)} 篇 | 语料 {len(docs)} 段 | 评测 {len(qas)} 题 | "
        f"检索器={retriever_kind} | max_rounds={max_rounds} | "
        f"memory={memory_mode} | turns={turns}",
        flush=True,
    )
    if memory_mode != "off" and retriever_kind != "engine":
        print("注意：自进化记忆需要 --retriever engine（稳定 doc id + 门控），本次记忆不会生效", flush=True)

    cfg = RagConfig().with_overrides(
        base_url=args.base_url,
        model=args.model,
        top_k=args.top_k,
        retrieval_mode="engine" if retriever_kind == "engine" else "vector",
        max_retrieval_rounds=max_rounds,
        mock=True if args.mock else None,
        sub_workflow_override="" if args.no_sub_override else None,
        memory_feedback=memory_mode,
        memory_strategies=not getattr(args, "no_strategies", False),
    )
    if getattr(args, "memory_db", ""):
        cfg = cfg.with_overrides(memory_path=_tagged_db_path(args.memory_db, memory_tag))
    # 三档对比时把所有 Agent 温度固定为同一值（通常 0），使各档共享同一批 plan，
    # 避免 Planner 采样差异把「引擎/循环」的增益与「分解率波动」混淆。
    if getattr(args, "temperature", None) is not None:
        cfg.agent_temperatures = {name: float(args.temperature) for name in AGENT_NAMES}
        cfg.default_temperature = float(args.temperature)
    llm = build_llm(cfg)

    # 检索器：open=全局语料（问题里带上论文标题）；paper=每篇论文独立语料
    retrievers: Dict[str, Any] = {}
    if args.scope == "open":
        print("构建全局索引 ...", flush=True)
        retrievers["__all__"] = _make_retriever(args, docs, retriever_kind, verbose=True)
    else:
        print("构建逐论文索引 ...", flush=True)
        for pid, paper in papers.items():
            sub_docs = [d for d, m in zip(docs, meta) if m["paper_id"] == pid]
            retrievers[pid] = _make_retriever(args, sub_docs, retriever_kind)

    # Runtime/graph 只建一次：记忆库跨轮持久，多轮协议（GAM-RAG k-turn memorization）
    # 全靠同一批 Runtime 实例在轮间累积经验。
    runtimes: Dict[str, Runtime] = {}
    graphs = {}
    for key, retriever in retrievers.items():
        rt_cfg = cfg
        if args.scope == "paper" and cfg.memory_path:
            # 逐论文语料的 doc id 各自独立，记忆库必须按论文隔离
            rt_cfg = cfg.with_overrides(memory_path=_tagged_db_path(cfg.memory_path, str(key)))
        runtimes[key] = Runtime(cfg=rt_cfg, llm=llm, retriever=retriever)
        graphs[key] = build_rag_graph(runtimes[key])

    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def worker(item: Dict[str, Any], turn: int) -> Dict[str, Any]:
        key = item["paper_id"] if args.scope == "paper" else "__all__"
        question = item["question"]
        if args.scope == "open":
            question = f'In the paper titled "{item["title"]}": {question}'
        async with semaphore:
            state = await graphs[key].ainvoke(
                {
                    "question": question,
                    "gold_answer": item["gold"],
                    "gold_evidence": item["evidence"],
                    "trace": [],
                }
            )
        row = evaluate_one(item, state)
        row["turn"] = turn
        return row

    turn_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for turn in range(1, turns + 1):
        done = 0
        # P2.2 turn 内记忆隔离：本轮所有题看到同一份「轮初」记忆，写入暂存轮末提交
        for rt in runtimes.values():
            mem = getattr(rt, "memory", None)
            if mem is not None and hasattr(mem, "begin_defer"):
                mem.begin_defer()
        if turns > 1:
            print(f"---- 第 {turn}/{turns} 轮 ----", flush=True)

        async def counted(item: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal done
            row = await worker(item, turn)
            done += 1
            if done % 5 == 0 or done == len(qas):
                print(f"  进度 {done}/{len(qas)}", flush=True)
            return row

        rows = list(await asyncio.gather(*[counted(item) for item in qas]))
        for rt in runtimes.values():
            mem = getattr(rt, "memory", None)
            if mem is not None and hasattr(mem, "commit_defer"):
                mem.commit_defer()
        summary = summarize(rows)
        summary["turn"] = turn
        summary["memory"] = _memory_stats(runtimes)
        turn_summaries.append(summary)
        all_rows.extend(rows)
        if turns > 1:
            mem = summary["memory"]
            print(
                f"  turn {turn}: F1={summary.get('answer_f1', 0):.4f} "
                f"hitR={summary.get('evidence_hit_at_retrieval', 0):.4f} "
                f"rounds={summary.get('mean_retrieval_times', 0):.2f} "
                f"mem={mem.get('n_docs', 0)}段/π̄={mem.get('mean_pi', 0):.2f}"
                f"/+{int(mem.get('pos_updates', 0))}/-{int(mem.get('neg_updates', 0))}"
                f" | 策略={mem.get('n_strategies', 0)}条/ws={summary.get('warm_starts', 0)}",
                flush=True,
            )

    return {
        "summary": turn_summaries[-1],
        "turns": turn_summaries,
        "rows": all_rows,
        "config": vars(args),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QASPER 端到端评测 LangGraph 工作流")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--papers", type=int, default=12, help="参与构建语料的论文数")
    parser.add_argument("--questions", type=int, default=24, help="评测问题数")
    parser.add_argument("--answer-type", choices=["short", "yesno", "all"], default="short")
    parser.add_argument("--max-gold-words", type=int, default=20, help="short 模式下的金标词数上限")
    parser.add_argument("--scope", choices=["open", "paper"], default="open")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="model/Qwen3.8-27B-FP8")
    parser.add_argument("--embed-url", default="http://127.0.0.1:10200/v1")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--embed-batch", type=int, default=16)
    parser.add_argument(
        "--retriever",
        choices=["vector", "engine"],
        default="vector",
        help="vector=纯向量；engine=Corpus Interaction Engine（向量+BM25+entity+整形）",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="运行期检索循环上限：1=单次检索；>1 启用「检索→判断→不足→重写→再检索」",
    )
    parser.add_argument(
        "--tiers",
        action="store_true",
        help="一次性跑三档 A/B：{纯向量} / {+引擎} / {+引擎+循环}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="覆盖所有 Agent 的温度（三档对比建议用 0，使各档共享同一批 plan）",
    )
    parser.add_argument(
        "--memory",
        choices=["off", "gold", "agent"],
        default="off",
        help="自进化记忆（GAM-RAG）：off=关闭；gold=金标 evidence 标注反馈（评测口径）；"
        "agent=judge 排除/DS 保留标注反馈（生产口径）。需 --retriever engine",
    )
    parser.add_argument(
        "--memory-db",
        default="",
        help="记忆库 SQLite 持久化路径（跨进程进化）；空=仅进程内（多轮评测够用）",
    )
    parser.add_argument(
        "--no-strategies",
        action="store_true",
        help="消融：禁用 P1 策略记忆 warm-start（仅保留段落级文档记忆）",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="多轮记忆协议：同一题库重复评测 N 轮，记忆跨轮累积（GAM-RAG k-turn memorization）",
    )
    parser.add_argument("--mock", action="store_true", help="用离线定式模型自测评测脚本")
    parser.add_argument("--no-sub-override", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.tiers:
        tiers = [
            ("纯向量 (vector, rounds=1)", "vector", 1),
            ("+引擎 (engine, rounds=1)", "engine", 1),
            (f"+引擎+循环 (engine, rounds={args.max_rounds})", "engine", args.max_rounds),
        ]
        summaries: Dict[str, Dict[str, Any]] = {}
        tier_turns: Dict[str, List[Dict[str, Any]]] = {}
        for label, kind, rounds in tiers:
            print(f"\n########## 档位：{label} ##########", flush=True)
            result = asyncio.run(
                run_eval(args, retriever_kind=kind, max_rounds=rounds, memory_tag=f"{kind}-r{rounds}")
            )
            summaries[label] = result["summary"]
            tier_turns[label] = result["turns"]
            if args.out:
                Path(args.out).write_text(
                    json.dumps({"tiers": summaries, "tier_turns": tier_turns}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        keys = [
            "n_questions",
            "answer_f1",
            "answer_em",
            "evidence_hit_at_retrieval",
            "evidence_hit_at_generator",
            "format_valid_rate",
            "mean_steps_per_question",
            "mean_turn_number",
            "mean_retrieval_times",
        ]
        print("\n================ 三档对比 ================")
        print(f"{'metric':<30}" + "".join(f"{label:>26}" for label in summaries))
        for key in keys:
            print(f"{key:<30}" + "".join(f"{summaries[l].get(key, 0):>26.4f}" for l in summaries))
        if args.out:
            print(f"\n结果已写入 {args.out}")
        return 0

    result = asyncio.run(run_eval(args))

    if args.turns > 1:
        keys = [
            "answer_f1",
            "evidence_hit_at_retrieval",
            "evidence_hit_at_generator",
            "mean_retrieval_times",
            "mean_steps_per_question",
            "warm_starts",
        ]
        print("\n================ 逐轮进化曲线 ================")
        print(f"{'metric':<30}" + "".join(f"{'turn ' + str(t['turn']):>12}" for t in result["turns"]))
        for key in keys:
            print(f"{key:<30}" + "".join(f"{t.get(key, 0):>12.4f}" for t in result["turns"]))
        mems = [t.get("memory") or {} for t in result["turns"]]
        if any(mems):
            print(f"{'memory n_docs':<30}" + "".join(f"{m.get('n_docs', 0):>12}" for m in mems))
            print(f"{'memory mean_pi':<30}" + "".join(f"{m.get('mean_pi', 0):>12.3f}" for m in mems))
            print(f"{'memory n_strategies':<30}" + "".join(f"{m.get('n_strategies', 0):>12}" for m in mems))

    print("\n================ 汇总（最后一轮） ================")
    for key, value in result["summary"].items():
        print(f"  {key:<38} {value}")

    print("\n================ 逐题（前 12）================")
    for row in result["rows"][:12]:
        print(
            f"[{row['workflow_cate']:<4}] F1={row['final_f1']:.2f} EM={row['em']:.2f} "
            f"hitR={int(row['hit_retrieval'])} hitAG={int(row['hit_generator'])} | "
            f"Q: {row['question'][:70]}"
        )
        print(f"        gold={row['gold'][:70]!r}")
        print(f"        pred={row['pred'][:70]!r}")
        if row["invalid_examples"]:
            print(f"        bad ={row['invalid_examples']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
