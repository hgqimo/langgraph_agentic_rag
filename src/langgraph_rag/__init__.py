"""基于 LangGraph 的 agentic RAG 工作流。

参考实现源码映射：
- qa_manager/base_agent_tag.py             -> agents.py（prompt + 严格解析）
- qa_manager/qa.py                         -> runtime.py / state.py
- qa_manager/{compute_metrics,normalize_answers}.py -> reward.py
- verl/trainer/ppo/ray_trainer.py::async_rollout_single -> graph.py（图拓扑）

快速开始::

    from langgraph_rag import RagConfig, Runtime, build_llm, build_rag_graph

    cfg = RagConfig(mock=True, mock_route="qds")
    graph = build_rag_graph(Runtime(cfg=cfg, llm=build_llm(cfg)))
    state = graph.invoke({"question": "...", "gold_answer": "...", "trace": []})
    print(state["answer"], state["final_metrics"])
"""

from .config import AGENT_NAMES, RagConfig
from .graph import build_chain_subgraph, build_rag_graph, sanitize_chain
from .llm import ScriptedChatModel, build_llm
from .reward import compute_final_reward, compute_scores, summarize_metrics
from .runtime import Runtime
from .state import ChainState, RagState, SubQuery, TraceStep
from .tools import RetrievalTool

__all__ = [
    "AGENT_NAMES",
    "ChainState",
    "RagConfig",
    "RagState",
    "RetrievalTool",
    "Runtime",
    "ScriptedChatModel",
    "SubQuery",
    "TraceStep",
    "build_chain_subgraph",
    "build_llm",
    "build_rag_graph",
    "compute_final_reward",
    "compute_scores",
    "sanitize_chain",
    "summarize_metrics",
]
