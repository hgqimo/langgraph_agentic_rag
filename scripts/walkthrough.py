"""单条 query 的工作流逐步演练（观察每个节点做了什么、每个 Agent 的 prompt/输出/解析）。

用 LangGraph 的 `stream(stream_mode=["updates", "values"])`：
- updates: 每个节点本次的状态增量（含该节点内发生的 Agent 调用 trace）
- values : 累积后的完整状态快照，最后一个即最终状态

用法::

    # 从 QASPER 取第 K 篇论文的第 J 个可评测问题（paper 范围=该论文自身段落语料）
    python scripts/walkthrough.py --qasper-data datasets/qasper/raw/qasper-dev-v0.3.json \
        --papers 20 --paper-rank 3 --qa-rank 0 --scope paper \
        --base-url http://127.0.0.1:8000/v1 --model model/Qwen3.8-27B-FP8 \
        --embed-url http://127.0.0.1:10200/v1

    # 直接给问题（检索走 mock/llm/none/http 或 --corpus 指定的本地向量语料）
    python scripts/walkthrough.py --question "..." --gold "..." --retrieval-mode llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):  # 支持不经安装直接运行
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langgraph_rag.config import RagConfig  # noqa: E402
from langgraph_rag.graph import build_rag_graph  # noqa: E402
from langgraph_rag.llm import build_llm  # noqa: E402
from langgraph_rag.runtime import Runtime  # noqa: E402

MODULE_OF_AGENT = {
    "QueryRewriteAgent": "QR",
    "DocumentSelectionAgent": "DS",
    "AnswerGenerationAgent": "AG",
    "RetrievalTool": "R",
    "EvidenceJudgeAgent": "J",
}


def _clip(text: Any, limit: int) -> str:
    text = str(text)
    text = text.replace("\r", "")
    if limit and len(text) > limit:
        return text[:limit] + f" ...(+{len(text) - limit} chars)"
    return text


def _abbrev(value: Any, limit: int = 120) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"list[{len(value)}] " + _clip(value[0], limit)
        return _clip(value, limit)
    if isinstance(value, dict):
        return _clip({k: v for k, v in value.items() if k != "trace"}, limit)
    return _clip(value, limit)


def print_messages(messages: List[Dict[str, str]], max_chars: int, full: bool) -> None:
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content", "")
        if role == "system" and not full:
            first_line = content.split("\n")[0]
            print(f"      [system] ({len(content)} chars) {_clip(first_line, 140)}")
        else:
            indent = "      "
            body = _clip(content, 0 if full else max_chars).replace("\n", f"\n{indent}")
            print(f"      [{role}] {body}")


def print_step(step: Dict[str, Any], args: argparse.Namespace) -> None:
    module = MODULE_OF_AGENT.get(step["agent"], "")
    flag = "OK " if step.get("is_valid") else "非法"
    print(
        f"\n    [模块 {module or '-':<2}] Agent={step['agent']}  解析={flag} "
        f"penalty={step['penalty']:+.1f}"
    )
    print("    prompt:")
    print_messages(step.get("messages") or [], args.max_prompt_chars, args.full_prompts)
    print(f"    原始输出: {_clip(step.get('output', ''), args.max_output_chars)}")
    extra = step.get("extra") or {}
    if extra:
        print(f"    附加: {_abbrev(extra, 160)}")


def print_node(node: str, update: Dict[str, Any], args: argparse.Namespace) -> None:
    print(f"\n{'=' * 100}\n### 节点 [{node}]")
    for key, value in update.items():
        if key == "trace":
            continue
        print(f"    state.{key} = {_abbrev(value)}")
    for step in update.get("trace") or []:
        print_step(step, args)


def build_setup(args: argparse.Namespace):
    """返回 (cfg, question, gold, retriever, 补充说明)。"""
    from langgraph_rag.vector import VectorRetriever

    corpus: Optional[List[str]] = None
    question, gold = args.question, args.gold
    note = ""

    if args.qasper_data:
        from eval.qasper_eval import build_corpus, build_qas, load_papers

        papers = load_papers(args.qasper_data, args.papers, args.seed)
        docs, meta = build_corpus(papers)
        qas = build_qas(papers, "short", args.max_gold_words)
        paper_ids = list(papers)
        pid = paper_ids[args.paper_rank]
        paper_qas = [q for q in qas if q["paper_id"] == pid]
        item = paper_qas[args.qa_rank]
        question, gold = item["question"], item["gold"]
        title = item["title"]
        if args.scope == "paper":
            corpus = [d for d, m in zip(docs, meta) if m["paper_id"] == pid]
            note = f"论文《{title}》自身的 {len(corpus)} 段作为语料（闭域）"
        else:
            corpus = docs
            question = f'In the paper titled "{title}": {question}'
            note = f"全局 {len(corpus)} 段语料（开域，问题里带上论文标题）"

    cfg = RagConfig().with_overrides(
        base_url=args.base_url,
        model=args.model,
        retrieval_mode="engine" if args.engine else args.retrieval_mode,
        top_k=args.top_k,
        max_retrieval_rounds=args.max_rounds,
        mock=True if args.mock else None,
        sub_workflow_override="" if args.no_sub_override else None,
    )

    def make_retriever(docs):
        if args.engine:
            from langgraph_rag.engine import CorpusInteractionEngine

            semantic = VectorRetriever(
                docs=docs,
                base_url=args.embed_url,
                model=args.embed_model,
                batch_size=args.embed_batch,
                verbose=False,
            )
            return CorpusInteractionEngine(list(docs), semantic).build()
        return VectorRetriever(
            docs=docs,
            base_url=args.embed_url,
            model=args.embed_model,
            batch_size=args.embed_batch,
            verbose=False,
        ).build()

    retriever = None
    if corpus is not None:
        retriever = make_retriever(corpus)
    elif args.corpus:
        retriever = make_retriever(VectorRetriever.from_jsonl(args.corpus).docs)
    return cfg, question, gold, retriever, note


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="单条 query 的 LangGraph Agentic RAG 工作流逐步演练")
    parser.add_argument("--question", default=None)
    parser.add_argument("--gold", default="")
    parser.add_argument("--qasper-data", default=None, help="从 QASPER 取例题")
    parser.add_argument("--papers", type=int, default=20)
    parser.add_argument("--paper-rank", type=int, default=0, help="取第 K 篇论文")
    parser.add_argument("--qa-rank", type=int, default=0, help="该论文的第 J 个可评测问题")
    parser.add_argument("--scope", choices=["paper", "open"], default="paper")
    parser.add_argument("--max-gold-words", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="model/Qwen3.8-27B-FP8")
    parser.add_argument("--embed-url", default="http://127.0.0.1:10200/v1")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--embed-batch", type=int, default=16)
    parser.add_argument("--corpus", default=None, help="本地向量语料 JSONL")
    parser.add_argument(
        "--retrieval-mode",
        choices=["auto", "http", "mock", "llm", "none", "vector", "engine"],
        default="vector",
    )
    parser.add_argument(
        "--engine",
        action="store_true",
        help="用 Corpus Interaction Engine（向量+BM25 融合+entity+include/exclude）替代纯向量检索",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="运行期检索循环上限：1=单次检索，>1 启用「检索→判断→不足→重写→再检索」",
    )
    parser.add_argument("--top-k", type=int, default=5, help="R 模块返回文档数")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-sub-override", action="store_true")

    parser.add_argument("--max-prompt-chars", type=int, default=600)
    parser.add_argument("--max-output-chars", type=int, default=400)
    parser.add_argument("--full-prompts", action="store_true", help="打印完整 prompt")
    args = parser.parse_args(argv)

    cfg, question, gold, retriever, note = build_setup(args)
    if not question:
        parser.error("请提供 --question 或 --qasper-data")

    print("=" * 100)
    print(f"模型: {cfg.model} | 检索: {cfg.resolved_retrieval_mode()} | top_k={cfg.top_k} | 思考模式: "
          f"{'关' if cfg.disable_thinking else '开'}")
    if note:
        print(f"语料: {note}")
    print(f"query : {question}")
    print(f"gold  : {gold}")
    print("=" * 100)

    graph = build_rag_graph(Runtime(cfg=cfg, llm=build_llm(cfg), retriever=retriever))
    initial = {"question": question, "gold_answer": gold, "trace": []}

    final_state: Dict[str, Any] = {}
    for mode, chunk in graph.stream(initial, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                print_node(node, update, args)
        else:
            final_state = chunk

    metrics = final_state.get("final_metrics") or {}
    print(f"\n{'=' * 100}\n### 最终结果")
    print(f"    answer        = {final_state.get('answer')!r}   (gold={gold!r})")
    print(f"    workflow      = {final_state.get('workflow')!r}  cate={final_state.get('workflow_cate')}")
    print(f"    final_reward  = {final_state.get('final_reward')}")
    print(f"    turn_number   = {metrics.get('turn_number')}  retrieval_times = {metrics.get('retrieval_times')}")

    cost = (cfg.alpha_turn_penalty * min(max((metrics.get("turn_number") or 1) - 1, 0), 3) / 3
            + cfg.beta_retrieval_penalty * min(max(metrics.get("retrieval_times") or 0, 0), 3) / 3)
    print(f"\n    每步奖励构成（对齐训练侧口径：whole = penalty + final_reward，再减去 turn/检索成本）")
    for agent, penalties in (metrics.get("step_reward") or {}).items():
        if not penalties:
            continue
        row = ", ".join(
            f"pen={p:+.0f} whole={w:+.2f}" for p, w in zip(metrics["penalty"][agent], penalties)
        )
        print(f"      {agent:<32} x{len(penalties)}  {row}   (成本项 {cost:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
