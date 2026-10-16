"""Multi-hop QA（HotpotQA / 2WikiMultihopQA / MuSiQue）自进化评测。

对齐 GAM-RAG 论文协议：

- **k-turn memorization**：同一题库重复评测 k 轮，记忆跨轮累积，报告逐轮曲线
  与 k-turn 相对 0-turn 的提升（论文 Table 1 的 ↑% 口径）；
- **Effectiveness 指标**：
  - Judge-Acc：LLM judge 判定预测答案与金标是否一致（论文用 GPT-4o，
    这里按用户要求用**自己的骨干模型**，temperature=0）；
  - Contain-Acc：预测答案是否包含金标（含 MuSiQue 的 answer_aliases）；
  - token-level F1 / EM（本仓库奖励口径）；
  - evidence_hit_at_retrieval / at_generator（检索侧命中，金标 supporting 标注）；
- **Efficiency 指标**：mean_retrieval_times / mean_steps（论文报告检索时间与
  token 消耗，这里用循环轮数与步数代理）；
- **三臂**：off（自进化前基线）/ agent（反馈来自 judge/DS 自我标注，生产口径）
  / gold（反馈来自 GT，评测口径）。off 臂只跑 1 轮（记忆关闭时多轮曲线平坦，
  §9.4 已验证），agent/gold 跑满 k 轮。

用法::

    python eval/multihop_eval.py --dataset musique --questions 60 --turns 5 \
        --base-url http://127.0.0.1:8000/v1 --model model/Qwen3.8-27B-FP8 \
        --workers 4 --temperature 0 --arms off,agent,gold
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import convert_to_messages

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:      # 支持不经安装直接运行
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:              # 使 eval.qasper_eval 可作为包导入
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph_rag.config import AGENT_NAMES, RagConfig  # noqa: E402
from langgraph_rag.graph import build_rag_graph  # noqa: E402
from langgraph_rag.llm import build_llm, text_of  # noqa: E402
from langgraph_rag.runtime import Runtime  # noqa: E402

from eval.qasper_eval import (  # noqa: E402
    _make_retriever,
    _memory_stats,
    evaluate_one,
    summarize,
)

# 数据根目录：默认仓库 datasets/，可用 RAG_DATA_ROOT 覆盖
_DATA_ROOT = os.environ.get("RAG_DATA_ROOT") or str(_REPO_ROOT / "datasets")
DEFAULT_PATHS = {
    "hotpotqa": f"{_DATA_ROOT}/HotpotQA/raw/hotpot_dev_distractor_v1.json",
    "2wiki": f"{_DATA_ROOT}/2WikiMultihopQA/dev.json",
    "musique": f"{_DATA_ROOT}/MuSiQue/musique_ans_v1.0_dev.jsonl",
}


# ---------------------------------------------------------------------------- 数据适配
def _load_raw(dataset: str, path: str) -> List[Dict[str, Any]]:
    if dataset == "musique":
        return [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    return json.load(open(path, "r", encoding="utf-8"))


def _norm_text(text: str) -> str:
    return " ".join(str(text).split()).lower()


def load_multihop(
    dataset: str, path: str, n_questions: int, seed: int
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]]]:
    """采样 n_questions 题，合并题目自带 context 为共享检索语料（HippoRAG2 协议）。

    返回 (docs, meta, qas)；qas 元素与 qasper_eval.evaluate_one 兼容：
    {question, gold, kind, evidence, aliases}。仅保留有 supporting 标注的题
    （gold 臂反馈需要金标 evidence）。
    """
    raw = _load_raw(dataset, path)
    rng = random.Random(seed)
    rng.shuffle(raw)

    docs: List[str] = []
    meta: List[Dict[str, str]] = []
    seen: Dict[Tuple[str, str], int] = {}
    qas: List[Dict[str, Any]] = []

    def add_context(title: str, text: str) -> None:
        key = (title, text)
        if key in seen:
            return
        seen[key] = len(docs)
        docs.append(f"[{title}] {text}")
        meta.append({"title": title, "raw": text})

    for item in raw:
        if len(qas) >= n_questions:
            break
        if dataset in ("hotpotqa", "2wiki"):
            question = str(item.get("question") or "").strip()
            gold = str(item.get("answer") or "").strip()
            context = item.get("context") or []
            sup = [tuple(x) for x in (item.get("supporting_facts") or [])]
            if not question or not gold or not context or not sup:
                continue
            sent_by_title: Dict[str, List[str]] = {}
            for title, sents in context:
                sent_by_title[str(title)] = [str(s) for s in (sents or [])]
                add_context(str(title), " ".join(str(s) for s in (sents or [])).strip())
            evidence = []
            for title, sid in sup:
                sents = sent_by_title.get(str(title)) or []
                if isinstance(sid, int) and 0 <= sid < len(sents):
                    evidence.append(sents[sid].strip())
            if not evidence:
                continue
            qas.append({
                "question": question,
                "gold": gold,
                "kind": str(item.get("type") or "unknown"),
                "evidence": [e for e in evidence if e],
                "aliases": [gold],
            })
        elif dataset == "musique":
            question = str(item.get("question") or "").strip()
            gold = str(item.get("answer") or "").strip()
            paragraphs = item.get("paragraphs") or []
            if not question or not gold or not paragraphs:
                continue
            evidence: List[str] = []
            for para in paragraphs:
                title = str(para.get("title") or "")
                text = str(para.get("paragraph_text") or "").strip()
                if not text:
                    continue
                add_context(title, text)
                if para.get("is_supporting"):
                    evidence.append(text)
            if not evidence:
                continue
            aliases = [gold] + [str(a).strip() for a in (item.get("answer_aliases") or []) if str(a).strip()]
            qas.append({
                "question": question,
                "gold": gold,
                "kind": str(item.get("id") or "x").split("__")[0],
                "evidence": evidence,
                "aliases": aliases,
            })
        else:
            raise ValueError(f"未知数据集：{dataset}")

    return docs, meta, qas


# ---------------------------------------------------------------------------- GAM-RAG 指标
def contain_acc(pred: str, gold: str, aliases: List[str]) -> float:
    """Contain-Match Accuracy：预测答案是否包含金标（或其别名）。"""
    p = _norm_text(pred)
    if not p:
        return 0.0
    for cand in [gold, *(aliases or [])]:
        c = _norm_text(cand)
        if c and c in p:
            return 1.0
    return 0.0


_JUDGE_SYSTEM = (
    "You are a strict answer-match judge for a QA benchmark. Decide whether the "
    "predicted answer matches the ground truth answer semantically (same entity, "
    "fact or polarity; ignore extra wording, articles and formatting). "
    "Respond with exactly one word: YES or NO."
)


async def judge_batch(
    llm: Any, rows: List[Dict[str, Any]], semaphore: asyncio.Semaphore,
    cache: Dict[Tuple[str, str], float],
) -> None:
    """用自有骨干模型做 LLM judge（论文用 GPT-4o 的替代），结果写入 row['judge_acc']。"""
    bound = llm.bind(temperature=0)

    async def one(row: Dict[str, Any]) -> None:
        pred, gold = str(row.get("pred") or ""), str(row.get("gold") or "")
        aliases = tuple(sorted(row.get("aliases") or []))
        if not _norm_text(pred):
            row["judge_acc"] = 0.0
            return
        key = (_norm_text(pred), _norm_text(gold) + "|" + "|".join(aliases))
        if key in cache:
            row["judge_acc"] = cache[key]
            return
        messages = convert_to_messages([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {row.get('question', '')}\n"
                    f"Ground truth: {gold}\n"
                    f"Acceptable aliases: {', '.join(a for a in aliases if a != gold) or '(none)'}\n"
                    f"Predicted answer: {pred}\n"
                    "Does the predicted answer match the ground truth? (YES/NO)"
                ),
            },
        ])
        async with semaphore:
            try:
                out = _norm_text(text_of(await bound.ainvoke(messages)))
            except Exception as exc:  # noqa: BLE001
                logging.warning("judge 调用失败（记 0）：%s", exc)
                out = ""
        val = 1.0 if out.startswith("yes") else 0.0
        cache[key] = val
        row["judge_acc"] = val

    await asyncio.gather(*[one(r) for r in rows])


# ---------------------------------------------------------------------------- 单臂运行
async def run_arm(
    args: argparse.Namespace, arm: str, docs: List[str], qas: List[Dict[str, Any]],
    llm: Any,
) -> Dict[str, Any]:
    turns = 1 if arm == "off" else max(1, args.turns)
    cfg = RagConfig().with_overrides(
        base_url=args.base_url,
        model=args.model,
        top_k=args.top_k,
        retrieval_mode="engine",
        max_retrieval_rounds=args.max_rounds,
        memory_feedback=arm,
        memory_strategies=not args.no_strategies,
    )
    if getattr(args, "temperature", None) is not None:
        cfg.agent_temperatures = {name: float(args.temperature) for name in AGENT_NAMES}
        cfg.default_temperature = float(args.temperature)

    print(f"\n########## 臂：{arm}（turns={turns}）##########", flush=True)
    retriever = _make_retriever(args, docs, "engine", verbose=True)
    runtime = Runtime(cfg=cfg, llm=llm, retriever=retriever)
    graph = build_rag_graph(runtime)

    semaphore = asyncio.Semaphore(max(1, args.workers))
    judge_sem = asyncio.Semaphore(8)
    judge_cache: Dict[Tuple[str, str], float] = {}

    async def worker(item: Dict[str, Any], turn: int) -> Dict[str, Any]:
        async with semaphore:
            state = await graph.ainvoke({
                "question": item["question"],
                "gold_answer": item["gold"],
                "gold_evidence": item["evidence"],
                "trace": [],
            })
        row = evaluate_one(item, state)
        row["turn"] = turn
        row["aliases"] = item.get("aliases") or []
        return row

    turn_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for turn in range(1, turns + 1):
        done = 0
        # P2.2 turn 内记忆隔离：本轮所有题看到同一份「轮初」记忆，写入暂存轮末提交
        mem = getattr(runtime, "memory", None)
        if mem is not None and hasattr(mem, "begin_defer"):
            mem.begin_defer()

        async def counted(item: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal done
            row = await worker(item, turn)
            done += 1
            if done % 10 == 0 or done == len(qas):
                print(f"  [{arm}] 进度 {done}/{len(qas)}", flush=True)
            return row

        rows = list(await asyncio.gather(*[counted(item) for item in qas]))
        if mem is not None and hasattr(mem, "commit_defer"):
            mem.commit_defer()
        await judge_batch(llm, rows, judge_sem, judge_cache)
        for row in rows:
            row["contain_acc"] = contain_acc(row.get("pred", ""), row.get("gold", ""), row.get("aliases") or [])
        summary = summarize(rows)
        summary["judge_acc"] = sum(r["judge_acc"] for r in rows) / len(rows) if rows else 0.0
        summary["contain_acc"] = sum(r["contain_acc"] for r in rows) / len(rows) if rows else 0.0
        summary["turn"] = turn
        summary["memory"] = _memory_stats({"_": runtime})
        turn_summaries.append(summary)
        all_rows.extend(rows)
        mem = summary["memory"]
        print(
            f"  [{arm}] turn {turn}: JudgeAcc={summary['judge_acc']:.4f} "
            f"ContainAcc={summary['contain_acc']:.4f} F1={summary.get('answer_f1', 0):.4f} "
            f"hitR={summary.get('evidence_hit_at_retrieval', 0):.4f} "
            f"rounds={summary.get('mean_retrieval_times', 0):.2f} "
            f"mem={mem.get('n_docs', 0)}段/π̄={mem.get('mean_pi', 0):.2f}"
            f" | 策略={mem.get('n_strategies', 0)}条/ws={summary.get('warm_starts', 0)}",
            flush=True,
        )

    return {"arm": arm, "turns": turn_summaries, "rows": all_rows}


# ---------------------------------------------------------------------------- 汇总
def print_comparison(dataset: str, results: Dict[str, Dict[str, Any]]) -> None:
    """GAM-RAG Table 1 风格：各臂末轮 vs 0-turn 基线（off turn1）的 ↑%。"""
    base = results.get("off", {}).get("turns", [{}])[0] if "off" in results else {}
    keys = [
        ("judge_acc", "Judge-Acc"),
        ("contain_acc", "Contain-Acc"),
        ("answer_f1", "F1"),
        ("evidence_hit_at_retrieval", "hitR"),
        ("evidence_hit_at_generator", "hitAG"),
        ("mean_retrieval_times", "rounds"),
        ("mean_steps_per_question", "steps"),
    ]
    rows_out: List[Tuple[str, Dict[str, Any]]] = []
    for arm, res in results.items():
        tlist = res.get("turns") or []
        if not tlist:
            continue
        rows_out.append((f"{arm}(t1)", tlist[0]))
        if len(tlist) > 1:
            rows_out.append((f"{arm}(t{tlist[-1]['turn']})", tlist[-1]))

    print(f"\n================ {dataset}：三臂对比（GAM-RAG 口径） ================")
    print(f"{'arm':<12}" + "".join(f"{label:>14}" for _, label in keys))
    for name, s in rows_out:
        cells = []
        for key, _ in keys:
            v = float(s.get(key, 0) or 0)
            cell = f"{v:.4f}"
            if base and name != "off(t1)" and key not in ("mean_retrieval_times", "mean_steps_per_question"):
                b = float(base.get(key, 0) or 0)
                cell += f"({'+' if v >= b else ''}{(v - b) * 100:.0f})"
            cells.append(cell)
        print(f"{name:<12}" + "".join(f"{c:>14}" for c in cells))

    # 效率相对变化（论文报告检索时间下降 %）
    if base:
        for name, s in rows_out:
            if name == "off(t1)":
                continue
            b = float(base.get("mean_retrieval_times", 0) or 0)
            v = float(s.get("mean_retrieval_times", 0) or 0)
            if b > 0:
                print(f"  {name}: retrieval time {(v - b) / b * 100:+.1f}% vs 0-turn")


async def run_all(args: argparse.Namespace) -> int:
    dataset = args.dataset
    path = args.data or DEFAULT_PATHS[dataset]
    docs, meta, qas = load_multihop(dataset, path, args.questions, args.seed)
    if not qas:
        print("没有可用的题（检查数据集路径与过滤条件）", flush=True)
        return 1
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(
        f"数据集 {dataset} | 语料 {len(docs)} 段（{len(meta)} 去重）| 评测 {len(qas)} 题 | "
        f"arms={arms} | turns={args.turns} | max_rounds={args.max_rounds}",
        flush=True,
    )

    cfg = RagConfig().with_overrides(base_url=args.base_url, model=args.model)
    llm = build_llm(cfg)

    results: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        results[arm] = await run_arm(args, arm, docs, qas, llm)

    print_comparison(dataset, results)
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "n_docs": len(docs),
                    "n_questions": len(qas),
                    "args": {k: v for k, v in vars(args).items()},
                    "arms": {
                        arm: {"turns": res["turns"], "rows": res["rows"]}
                        for arm, res in results.items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_PATHS))
    parser.add_argument("--data", default="", help="数据集路径（缺省用内置路径）")
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--turns", type=int, default=5, help="k-turn memorization 的 k（off 臂恒为 1）")
    parser.add_argument("--arms", default="off,agent,gold")
    parser.add_argument("--no-strategies", action="store_true", help="消融：禁用策略记忆 warm-start")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=RagConfig().base_url)
    parser.add_argument("--model", default="model/Qwen3.8-27B-FP8")
    parser.add_argument("--embed-url", default=RagConfig().embed_base_url)
    parser.add_argument("--embed-model", default=RagConfig().embed_model)
    parser.add_argument("--embed-batch", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=RagConfig().top_k)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
