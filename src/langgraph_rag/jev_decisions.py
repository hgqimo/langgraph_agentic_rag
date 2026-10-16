"""Jev 决策模型规范客户端：与 OpenAI Decisions API 同构的调用层。

背景（developers.openai.com/api/docs/guides/decisions，Decisions API public beta）：
OpenAI 用独立端点 ``POST /v1/decisions`` 做「类型化决策」——三种问题类型：

    predicate : 条件成立概率（0-1）                      -> answer.probability
    choice    : 固定选项选一个                            -> answer.choice + probabilities + confidence
    score     : 有序 levels 的概率加权平均                -> answer.score + probabilities + confidence

请求 ``{model, input, questions:[{type, name, instructions, choices|levels}]}``，
响应 ``{answers:[{type, name, ...}]}``，官方 SDK 形态为
``client.decisions.create(...)``（openai>=3.26.0，模型 gpt-6-luna）。

我们的 NeoHorse-Jev-4B（``neohorse-decision serve``，本地 GPU，prefill-only
零解码，零调用成本）是同一范式的本地实现（服务端自称 "System One wire
adapter"），类型一一对应、连 score 的「概率加权平均」定义与 confidence
统计算法都一致，差距只在 wire format：

    OpenAI                    NeoHorse-Jev-4B
    ------------------------  ------------------------------------------
    POST /v1/decisions        POST /v1/decision（questions 为 dict）
    input: str                state: str
    predicate                 noul（answer = p(true)）
    choices:[{value, desc}]   criteria: {label: desc}
    levels:[{label, desc}]    criteria: [desc, ...]（有序，低->高）
    answers: [...]            answers: {name: {...}}
    refusal                   （本地模型无拒答，不合成该语义）

本模块做双向翻译，对外暴露与官方 SDK 逐行同构的调用::

    client = JevDecisionsClient()                      # 默认 http://127.0.0.1:8080
    decision = client.decisions.create(
        input="I was charged twice for my order.",
        questions=[choice("department", "Which department should handle this?",
                          choices=[{"value": "billing", "description": "..."},
                                   {"value": "other", "description": "..."}])])
    for answer in decision.answers:
        if answer.type == "refusal": ...
        elif answer.type == "choice": print(answer.choice, answer.confidence)

将来若要把某类决策切到官方 Decisions API（gpt-6-luna，$0.10/1M input tokens），
只需把 client 换成 ``OpenAI(base_url=...).decisions``，业务代码不动
（官方 SDK 只能打官方端点，base_url 指向本地同样要求服务端 schema 兼容——
即使用官方 SDK 形态，本地 Jev 也走本客户端的翻译层）。

依赖：仅 requests（零内部依赖，可被 langgraph_rag 包与 jev_rerank 脚本共用）。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

#: 默认服务地址（与 rerank 既有默认一致）
JEV_URL = os.environ.get("RAG_JEV_URL", "http://127.0.0.1:8080")
#: Jev 模型名（服务端 MODEL_NAMES 别名集内任选其一）
JEV_MODEL = "NeoHorse-Jev-4B"

# ---------------------------------------------------------------- 问题构造（OpenAI 形态）


def predicate(name: str, instructions: str,
              on_true: Optional[str] = None,
              on_false: Optional[str] = None) -> Dict[str, Any]:
    """二元判断问题（OpenAI predicate；Jev 侧落为 noul，answer = p(条件为真)）。

    on_true/on_false 可给「条件成立/不成立」的细化描述（映射到 Jev 的
    noul criteria true/false 键），帮助模型对齐判定口径。
    """
    return {"type": "predicate", "name": name, "instructions": instructions,
            "on_true": on_true, "on_false": on_false}


def choice(name: str, instructions: str,
           choices: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """单选问题：choices = [{"value": ..., "description": ...}, ...]。

    建议带一个兜底选项（如 "other"），与官方指南一致。
    """
    if not 2 <= len(choices) <= 255:
        raise ValueError(f"choice 需要 2-255 个选项, got {len(choices)}")
    return {"type": "choice", "name": name, "instructions": instructions,
            "choices": [dict(c) for c in choices]}


def score(name: str, instructions: str,
          levels: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """有序评分问题：levels = [{"label": ..., "description": ...}, ...]，低->高。

    与 Jev 服务端 Score（2..10 级）对齐；answer.score 为概率加权平均
    （官方定义：Σ index·p，可落在两级之间）。
    """
    if not 2 <= len(levels) <= 10:
        raise ValueError(f"score 需要 2-10 个有序 levels, got {len(levels)}")
    return {"type": "score", "name": name, "instructions": instructions,
            "levels": [dict(lv) for lv in levels]}


# ---------------------------------------------------------------- 响应对象（OpenAI 形态）


@dataclass
class DecisionAnswer:
    """与官方 DecisionAnswer 同构：按 type 取相应字段。"""

    type: str                      # predicate | choice | score | refusal
    name: str
    probability: Optional[float] = None       # predicate
    choice: Optional[str] = None              # choice（argmax 的 value）
    score: Optional[float] = None             # score（概率加权平均）
    confidence: Optional[float] = None        # choice/score 的局部置信统计
    #: [{value, probability}]（choice）或 [{value, label, probability}]（score）
    probabilities: List[Dict[str, Any]] = field(default_factory=list)
    #: 原生 Jev 响应里的对应条目（审计/调试用）
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """与官方 Decision 同构：answers 数组，name 与请求 questions 对应。"""

    answers: List[DecisionAnswer] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def answer(self, name: str) -> DecisionAnswer:
        """按 question name 取答案（找不到抛 KeyError）。"""
        for a in self.answers:
            if a.name == name:
                return a
        raise KeyError(f"no answer named {name!r}")


# ---------------------------------------------------------------- 客户端


class JevDecisionsClient:
    """OpenAI Decisions API 同构的 Jev 客户端（内部走 POST /v1/decision）。

    与官方 SDK 的差异仅在构造：官方 ``OpenAI().decisions.create(...)``；
    这里 ``JevDecisionsClient().decisions.create(...)``，参数逐字同构。
    """

    def __init__(self, url: str = JEV_URL, model: str = JEV_MODEL,
                 timeout: float = 60.0, max_retries: int = 30,
                 endpoint: str = "/v1/decision"):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.endpoint = endpoint
        self.forward_calls = 0
        self.input_tokens = 0
        self.errors = 0
        self._ms_total = 0.0

    @property
    def decisions(self) -> "JevDecisionsClient":
        """对齐官方 SDK 的 ``client.decisions.create(...)`` 访问路径。"""
        return self

    # ------------------------------------------------------------ 调用
    def create(self, input: Union[str, Dict[str, Any], List[Any]],
               questions: Sequence[Dict[str, Any]],
               model: Optional[str] = None) -> Decision:
        """执行一次决策请求（零解码，prefill-only）。

        参数与官方 ``client.decisions.create(model=..., input=..., questions=...)``
        同构（model 有默认值故可省略）。失败抛异常——调用方按各自失败安全
        策略处理（如 rerank 在原序上透传）。
        """
        model = model or self.model
        wire_questions = {}
        for q in questions:
            qtype = str(q.get("type", "")).lower()
            name = str(q.get("name", "")).strip()
            if not name:
                raise ValueError(f"question 缺少 name: {q}")
            if name in wire_questions:
                raise ValueError(f"question name 重复: {name}")
            instructions = q.get("instructions", "")
            if qtype == "predicate":
                criteria = {}
                if q.get("on_true"):
                    criteria["true"] = q["on_true"]
                if q.get("on_false"):
                    criteria["false"] = q["on_false"]
                wire_questions[name] = {"type": "noul",
                                        "instructions": instructions,
                                        "criteria": criteria or None}
            elif qtype == "choice":
                opts = q.get("choices") or []
                if not 1 <= len(opts) <= 255:
                    raise ValueError(f"choice {name!r} 需要 1-255 个选项")
                wire_questions[name] = {
                    "type": "choice", "instructions": instructions,
                    "criteria": {str(c["value"]): c.get("description")
                                 for c in opts}}
            elif qtype == "score":
                levels = q.get("levels") or []
                if not 2 <= len(levels) <= 10:
                    raise ValueError(f"score {name!r} 需要 2-10 个 levels")
                wire_questions[name] = {
                    "type": "score", "instructions": instructions,
                    "criteria": [lv.get("description") for lv in levels]}
            else:
                raise ValueError(f"未知 question type: {qtype!r}"
                                 "（支持 predicate/choice/score）")

        body = {"model": model, "state": input, "questions": wire_questions}
        started = time.perf_counter()
        payload = self._post(body)
        latency_ms = (time.perf_counter() - started) * 1000
        self.forward_calls += 1
        self._ms_total += latency_ms
        self.input_tokens += int(payload.get("input_tokens", 0) or 0)
        return self._parse(payload, questions)

    # ------------------------------------------------------------ 响应翻译
    def _parse(self, payload: Dict[str, Any],
               questions: Sequence[Dict[str, Any]]) -> Decision:
        """Jev answers-dict -> OpenAI answers-list（按请求顺序）。"""
        raw_answers = payload.get("answers") or {}
        if not isinstance(raw_answers, dict):
            raise ValueError(f"异常 Jev 响应: answers 不是对象: {payload!r}")
        answers: List[DecisionAnswer] = []
        for q in questions:
            name = str(q.get("name", ""))
            if name not in raw_answers:
                raise ValueError(f"Jev 响应缺少 question {name!r} 的答案")
            answers.append(self._parse_answer(str(q.get("type", "")).lower(),
                                              name, raw_answers[name], q))
        return Decision(answers=answers, model=str(payload.get("model", "")),
                        input_tokens=int(payload.get("input_tokens", 0) or 0),
                        raw=payload)

    @staticmethod
    def _parse_answer(qtype: str, name: str, raw: Dict[str, Any],
                      q: Dict[str, Any]) -> DecisionAnswer:
        rtype = str(raw.get("type", "")).lower()
        if qtype == "predicate" or rtype == "noul":
            # noul answer = p(true) 即 predicate 的 probability
            prob = raw.get("noul")
            if prob is None:
                probs = raw.get("probabilities") or {}
                prob = probs.get("true", probs.get("1"))
            return DecisionAnswer(type="predicate", name=name,
                                  probability=float(prob), raw=raw)
        if rtype == "choice":
            probs = raw.get("probabilities") or {}
            return DecisionAnswer(
                type="choice", name=name, choice=raw.get("choice"),
                confidence=raw.get("confidence"),
                probabilities=[{"value": k, "probability": float(v)}
                               for k, v in probs.items()], raw=raw)
        if rtype == "score":
            probs = raw.get("probabilities") or {}
            labels = [lv.get("label") for lv in (q.get("levels") or [])]
            dist = []
            for key, value in probs.items():
                idx = int(key)
                dist.append({"value": idx,
                             "label": labels[idx] if idx < len(labels) else key,
                             "probability": float(value)})
            return DecisionAnswer(type="score", name=name,
                                  score=raw.get("score"),
                                  confidence=raw.get("confidence"),
                                  probabilities=dist, raw=raw)
        if rtype == "refusal":
            return DecisionAnswer(type="refusal", name=name, raw=raw)
        raise ValueError(f"未知 Jev answer type: {rtype!r} ({name})")

    # ------------------------------------------------------------ HTTP
    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """429 带 Retry-After 退避重试（与既有 JevReranker 一致）。"""
        for attempt in range(self.max_retries):
            resp = requests.post(f"{self.url}{self.endpoint}", json=body,
                                 timeout=self.timeout)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", 1))
                           + 0.05 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        self.errors += 1
        raise RuntimeError(f"Jev 服务持续 429 ({self.max_retries} 次退避)")

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).json().get(
                "status") == "ready"
        except Exception:  # noqa: BLE001
            return False

    @property
    def stats(self) -> Dict[str, Any]:
        return {"forward_calls": self.forward_calls,
                "input_tokens": self.input_tokens,
                "ms_per_call": (self._ms_total / self.forward_calls
                                if self.forward_calls else 0.0),
                "errors": self.errors}


# ---------------------------------------------------------------- 离线冒烟
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _FakeResp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    # 1) 三种问题类型的请求翻译 + 响应翻译（不依赖真实服务）
    captured = {}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResp({"model": "NeoHorse-Jev-4B", "input_tokens": 123,
                          "answers": {
                              "is_complaint": {"type": "noul", "noul": 0.93,
                                               "probabilities": {"false": 0.07,
                                                                 "true": 0.93}},
                              "department": {"type": "choice", "choice": "billing",
                                             "confidence": 0.91,
                                             "probabilities": {"billing": 0.95,
                                                               "technical": 0.03,
                                                               "other": 0.02}},
                              "severity": {"type": "score", "score": 1.1,
                                           "confidence": 0.55,
                                           "probabilities": {"0": 0.1, "1": 0.7,
                                                             "2": 0.2}}}})

    requests_post_real = requests.post
    requests.post = _fake_post  # type: ignore[assignment]
    try:
        client = JevDecisionsClient()
        decision = client.decisions.create(
            input="I was charged twice for my order and my phone won't turn on.",
            questions=[
                predicate("is_complaint", "Is this a billing complaint?",
                          on_true="Customer disputes a charge or refund.",
                          on_false="Not about charges."),
                choice("department", "Which department should handle this?",
                       choices=[{"value": "billing",
                                 "description": "Payments and refunds."},
                                {"value": "technical",
                                 "description": "Product problems."},
                                {"value": "other", "description": "Else."}]),
                score("severity", "How severe is this issue?",
                      levels=[{"label": "Cosmetic",
                               "description": "Appearance only."},
                              {"label": "Blocked",
                               "description": "A task fails, workaround exists."},
                              {"label": "Critical",
                               "description": "No workaround."}])])
    finally:
        requests.post = requests_post_real  # type: ignore[assignment]

    # 请求侧断言（OpenAI -> Jev wire）
    assert captured["url"].endswith("/v1/decision"), captured["url"]
    qs = captured["body"]["questions"]
    assert set(qs) == {"is_complaint", "department", "severity"}
    assert qs["is_complaint"]["type"] == "noul"
    assert qs["is_complaint"]["criteria"]["true"] == "Customer disputes a charge or refund."
    assert qs["department"]["type"] == "choice"
    assert qs["department"]["criteria"]["billing"] == "Payments and refunds."
    assert qs["severity"]["type"] == "score"
    assert qs["severity"]["criteria"][1] == "A task fails, workaround exists."
    assert captured["body"]["state"].startswith("I was charged twice")
    print("请求翻译 OK:", captured["body"]["questions"].keys())

    # 响应侧断言（Jev wire -> OpenAI 形态）
    a0 = decision.answer("is_complaint")
    assert a0.type == "predicate" and abs(a0.probability - 0.93) < 1e-9
    a1 = decision.answer("department")
    assert a1.type == "choice" and a1.choice == "billing"
    assert abs(a1.confidence - 0.91) < 1e-9
    assert a1.probabilities[0] == {"value": "billing", "probability": 0.95}
    a2 = decision.answer("severity")
    assert a2.type == "score" and abs(a2.score - 1.1) < 1e-9
    assert a2.probabilities[1] == {"value": 1, "label": "Blocked",
                                   "probability": 0.7}
    assert [a.name for a in decision.answers] == ["is_complaint", "department",
                                                  "severity"]
    print("响应翻译 OK:", [(a.name, a.type) for a in decision.answers])
    print("stats:", client.stats)
    print("ALL OK")
