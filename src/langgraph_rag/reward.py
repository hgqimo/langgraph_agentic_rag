"""LangGraph Agentic RAG：奖励与指标。

- normalize_answer / compute_scores：对齐 qa_manager/normalize_answers.py + compute_metrics.py
- summarize_metrics：对齐 verl/trainer/ppo/ray_trainer.py::async_rollout_single 中的
  record_metrics（reward / penalty / whole / times per agent）与 final_f1
- aggregate_metrics：对齐 async_rollout_batch 的批量均值统计
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence

from .config import AGENT_NAMES, RagConfig
from .state import TraceStep

YES_NO = {"yes", "Yes", "no", "No"}
YES_NO_NORMALIZED = {"yes", "no", "noanswer"}

_UNICODE_MAP = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
    "\u00a0": " ",
}


def normalize_answer(text: str, lowercase: bool = True) -> str:
    """小写 + 去标点 + 去冠词 + 折叠空白（SQuAD 风格）。"""
    text = str(text)
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    if lowercase:
        text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def compute_scores(predictions: Sequence[str], golds: Sequence[str]) -> Dict[str, float]:
    """逐样本累加后求均值；yes/no 类只计 acc/em，普通样本计入 f1/precision/recall。"""
    assert len(predictions) == len(golds), "预测答案和标准答案的长度不相等"

    final = {"acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    acc_em_count = 0
    fpr_count = 0

    for prediction, gold in zip(predictions, golds):
        golds_list = [gold] if isinstance(gold, str) else list(gold)
        temp = {"acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        norm_pred = normalize_answer(prediction)
        acc_em_count += 1

        if norm_pred in YES_NO_NORMALIZED and all(
            normalize_answer(g) in YES_NO_NORMALIZED for g in golds_list
        ):
            for g in golds_list:
                if norm_pred == normalize_answer(g):
                    temp["acc"] = temp["em"] = 1.0
            final["acc"] += temp["acc"]
            final["em"] += temp["em"]
            continue

        fpr_count += 1
        for g in golds_list:
            norm_gold = normalize_answer(g)
            if norm_gold and norm_gold in norm_pred:
                temp["acc"] = 1.0
            if norm_pred == norm_gold:
                temp["em"] = 1.0

            pred_tokens = norm_pred.split()
            gold_tokens = norm_gold.split()
            num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
            if num_same == 0:
                continue
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            temp["precision"] = max(precision, temp["precision"])
            temp["recall"] = max(recall, temp["recall"])
            temp["f1"] = max(2 * precision * recall / (precision + recall), temp["f1"])

        for key in ("acc", "em", "f1", "precision", "recall"):
            final[key] += temp[key]

    final["acc"] /= acc_em_count
    final["em"] /= acc_em_count
    if fpr_count:
        final["f1"] /= fpr_count
        final["precision"] /= fpr_count
        final["recall"] /= fpr_count
    else:
        final["f1"] = final["precision"] = final["recall"] = 0.0
    return final


def compute_final_reward(answer: str, gold_answer: str) -> float:
    """对齐 async_rollout_single：yes/no 用 EM，其余用 token-level F1。"""
    metric = "em" if gold_answer in YES_NO else "f1"
    return float(compute_scores([answer or ""], [gold_answer])[metric])


def scale_cost(n: int) -> float:
    """把 turn / retrieval 次数映射到 0~1（超过 3 记为 3）。"""
    return min(max(n, 0), 3) / 3.0


def summarize_metrics(
    trace: List[TraceStep],
    cfg: RagConfig,
    final_reward: float,
    turn_number: int,
    retrieval_count: int,
    workflow_cate: str,
) -> Dict[str, Any]:
    """按 Agent 汇总 reward / penalty / whole / times 与 step_reward。"""
    record: Dict[str, Any] = {
        "final_f1": final_reward,
        "turn_number": turn_number,
        "retrieval_times": retrieval_count,
        "workflow_cate": workflow_cate,
        "reward": {name: [] for name in AGENT_NAMES},
        "penalty": {name: [] for name in AGENT_NAMES},
        "whole": {name: [] for name in AGENT_NAMES},
        "step_reward": {name: [] for name in AGENT_NAMES},
        "times": {name: 0 for name in AGENT_NAMES},
    }

    cost = (
        cfg.alpha_turn_penalty * scale_cost(turn_number - 1)
        + cfg.beta_retrieval_penalty * scale_cost(retrieval_count)
    )
    for step in trace:
        agent = step.get("agent", "")
        if agent not in record["times"]:
            continue  # RetrievalTool 等非 LLM 步骤不计入 Agent 指标
        penalty = float(step.get("penalty", 0.0))
        whole = penalty + final_reward
        record["reward"][agent].append(final_reward)
        record["penalty"][agent].append(penalty)
        record["whole"][agent].append(whole)
        record["step_reward"][agent].append(whole - cost)
        record["times"][agent] += 1
    return record


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_metrics(records: List[Dict[str, Any]], prefix: str = "train_metrics") -> Dict[str, float]:
    """对齐 async_rollout_batch 的批量统计口径。"""
    if not records:
        return {}

    total = len(records)
    stats = {
        f"{prefix}/turn_number": _mean([r["turn_number"] for r in records]),
        f"{prefix}/retrieval_times": _mean([r["retrieval_times"] for r in records]),
        f"{prefix}/final_f1": _mean([r["final_f1"] for r in records]),
        f"{prefix}/qds_rate": sum(1 for r in records if r["workflow_cate"] == "QDS") / total,
        f"{prefix}/qdp_rate": sum(1 for r in records if r["workflow_cate"] == "QDP") / total,
        f"{prefix}/else_rate": sum(1 for r in records if r["workflow_cate"] == "ELSE") / total,
        f"{prefix}/avg_agent_calls": _mean(
            [sum(r["times"].values()) for r in records]
        ),
    }
    for metric in ("reward", "penalty", "whole", "step_reward", "times"):
        for agent in AGENT_NAMES:
            values: List[float] = []
            for r in records:
                value = r[metric][agent]
                # reward/penalty/whole/step_reward 是逐步列表，times 是标量
                values.extend(value if isinstance(value, list) else [value])
            stats[f"{prefix}/{metric}/{agent}"] = _mean(values)
    return stats
