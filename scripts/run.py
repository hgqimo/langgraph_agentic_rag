"""LangGraph Agentic RAG：命令行入口。

用法::

    # 1) 离线跑通（无需 LLM/检索服务，用于验证图拓扑与分支）
    python scripts/run.py --mock --mock-route qds \\
        --question "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?" \\
        --gold "mock final answer"

    # 2) 接真实 vLLM（OpenAI 兼容）+ 检索服务
    RAG_BASE_URL=http://127.0.0.1:8000/v1 RAG_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    RAG_SEARCH_URLS=http://10.148.12.65:8000/search \\
        python scripts/run.py --question "..." --show-process

    # 3) 本机 Qwen3 推理模型（思考模式默认关闭）+ 无检索服务时用模型自造候选证据
    python scripts/run.py --base-url http://127.0.0.1:8000/v1 \\
        --model model/Qwen3.8-27B-FP8 --retrieval-mode llm \\
        --question "..." --gold "..." --show-process

    # 4) 本地向量语料 + 运行期检索循环（Corpus Interaction Engine）
    #    max-rounds=1 即现行为；>1 启用「检索→判断→不足→重写→再检索」
    python scripts/run.py --retrieval-mode engine --corpus corpus.jsonl \\
        --max-rounds 3 --top-k 5 --question "..." --gold "..." --show-process

    # 5) 批量（JSONL，每行 {"question": ..., "answer": ...}），对应 async_rollout_batch
    python scripts/run.py --input data.jsonl --workers 8 --out result.json

    # 6) 打印流程图（Mermaid）
    python scripts/run.py --mermaid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:      # 支持不经安装直接运行
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from langgraph_rag.config import RagConfig  # noqa: E402
from langgraph_rag.graph import build_rag_graph  # noqa: E402
from langgraph_rag.llm import build_llm  # noqa: E402
from langgraph_rag.reward import aggregate_metrics  # noqa: E402
from langgraph_rag.runtime import Runtime  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph Agentic RAG 工作流")
    parser.add_argument("--question", type=str, default=None, help="单个问题")
    parser.add_argument("--gold", type=str, default="", help="金标答案（用于计算 EM/F1 奖励）")
    parser.add_argument("--input", type=str, default=None, help="批量输入 JSONL 文件")
    parser.add_argument("--limit", type=int, default=None, help="批量模式最多处理多少条")
    parser.add_argument("--workers", type=int, default=8, help="批量并发度")
    parser.add_argument("--out", type=str, default=None, help="把结果写入 JSON 文件")

    parser.add_argument("--mock", action="store_true", help="使用离线定式模型")
    parser.add_argument("--mock-route", choices=["qds", "qdp", "linear"], default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--search-url", type=str, default=None, help="逗号分隔的检索服务地址")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=["auto", "http", "mock", "llm", "none", "vector", "engine"],
        default=None,
        help="auto(默认)/http(真实检索服务)/mock/llm(本地模型自造证据)/none(不检索)"
        "/vector(本地纯向量)/engine(Corpus Interaction Engine)",
    )
    parser.add_argument(
        "--corpus", type=str, default=None, help="本地语料 JSONL（retrieval-mode=vector/engine 用）"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="运行期检索循环上限：1=单次检索(默认)，>1 启用「检索→判断→不足→重写→再检索」",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="保留思考模式；默认关闭（Qwen3 等推理模型需要显式 enable_thinking=false）",
    )
    parser.add_argument("--no-qdp", action="store_true", help="PlanningAgent 提示词中不提供 QDP")
    parser.add_argument("--no-sub-override", action="store_true", help="不覆盖 SubPlanningAgent 输出")
    parser.add_argument("--show-process", action="store_true", help="打印每个 Agent 的调用轨迹")
    parser.add_argument("--mermaid", action="store_true", help="打印图结构（Mermaid）")
    return parser.parse_args(argv)


def build_runtime(args: argparse.Namespace) -> Runtime:
    search_urls = [u.strip() for u in args.search_url.split(",")] if args.search_url else None
    cfg = RagConfig().with_overrides(
        mock=True if args.mock else None,
        mock_route=args.mock_route,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        search_urls=search_urls,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        corpus_path=args.corpus,
        max_retrieval_rounds=args.max_rounds,
        disable_thinking=False if args.think else None,
        enable_qdp=False if args.no_qdp else None,
        sub_workflow_override="" if args.no_sub_override else None,
    )
    return Runtime(cfg=cfg, llm=build_llm(cfg))


def _brief(messages: List[Dict[str, str]], max_len: int = 200) -> str:
    if not messages:
        return ""
    content = str(messages[-1].get("content", ""))
    content = " ".join(content.split())
    return content[:max_len] + ("..." if len(content) > max_len else "")


def print_trace(state: Dict[str, Any]) -> None:
    print("  --- Agent 轨迹 ---")
    for i, step in enumerate(state.get("trace") or [], 1):
        flag = "OK " if step.get("is_valid") else "BAD"
        print(
            f"  [{i:02d}] {step.get('agent'):<32} {flag} penalty={step.get('penalty'):>5} "
            f"| in: {_brief(step.get('messages') or [])}"
        )
        output = " ".join(str(step.get("output", "")).split())
        print(f"       out: {output[:200]}")


def print_result(state: Dict[str, Any], show_process: bool = False) -> None:
    metrics = state.get("final_metrics") or {}
    print(f"Q: {state.get('question')}")
    print(f"A: {state.get('answer')!r}")
    print(
        f"workflow={state.get('workflow')!r} cate={state.get('workflow_cate')} "
        f"turn={metrics.get('turn_number')} retrieval={metrics.get('retrieval_times')} "
        f"final_reward={metrics.get('final_f1')}"
    )
    times = metrics.get("times") or {}
    called = {k: v for k, v in times.items() if v}
    print(f"agent_calls={called}")
    penalties = {
        k: sum(v) for k, v in (metrics.get("penalty") or {}).items() if any(v)
    }
    if penalties:
        print(f"penalties={penalties}")
    if show_process:
        print_trace(state)


def run_single(graph, question: str, gold: str) -> Dict[str, Any]:
    return graph.invoke({"question": question, "gold_answer": gold or "", "trace": []})


async def run_batch(graph, items: List[Dict[str, str]], workers: int) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, workers))

    async def worker(item: Dict[str, str]) -> Dict[str, Any]:
        async with semaphore:
            return await graph.ainvoke(
                {
                    "question": item.get("question", ""),
                    "gold_answer": item.get("answer", "") or item.get("gold", ""),
                    "trace": [],
                }
            )

    return list(await asyncio.gather(*[worker(item) for item in items]))


def load_jsonl(path: str, limit: Optional[int]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    runtime = build_runtime(args)
    graph = build_rag_graph(runtime)

    if args.mermaid:
        print(graph.get_graph().draw_mermaid())
        if not args.question and not args.input:
            return 0

    if args.input:
        items = load_jsonl(args.input, args.limit)
        print(f"批量模式：{len(items)} 条，并发 {args.workers}，mock={runtime.cfg.mock}")
        states = asyncio.run(run_batch(graph, items, args.workers))
        records = [s.get("final_metrics") or {} for s in states]
        records = [r for r in records if r]
        summary = aggregate_metrics(records, prefix="batch_metrics")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if args.show_process and states:
            print_result(states[0], show_process=True)

        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    [
                        {
                            "question": s.get("question"),
                            "answer": s.get("answer"),
                            "gold": s.get("gold_answer"),
                            "workflow": s.get("workflow"),
                            "workflow_cate": s.get("workflow_cate"),
                            "final_reward": s.get("final_reward"),
                            "final_metrics": s.get("final_metrics"),
                            "trace": s.get("trace"),
                        }
                        for s in states
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"结果已写入 {args.out}")
        return 0

    question = args.question
    if not question:
        print("请提供 --question 或 --input（或用 --mermaid 仅打印图）", file=sys.stderr)
        return 2

    print(
        f"[config] model={runtime.cfg.model} base_url={runtime.cfg.base_url} "
        f"retrieval_mode={runtime.cfg.resolved_retrieval_mode()} "
        f"max_rounds={runtime.cfg.max_retrieval_rounds} "
        f"thinking={'on' if not runtime.cfg.disable_thinking else 'off'}"
    )
    state = run_single(graph, question, args.gold)
    print_result(state, show_process=args.show_process)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "question": state.get("question"),
                    "answer": state.get("answer"),
                    "gold": state.get("gold_answer"),
                    "workflow": state.get("workflow"),
                    "workflow_cate": state.get("workflow_cate"),
                    "sub_queries": state.get("sub_queries"),
                    "final_reward": state.get("final_reward"),
                    "final_metrics": state.get("final_metrics"),
                    "trace": state.get("trace"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
