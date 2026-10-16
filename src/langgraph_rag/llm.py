"""LangGraph Agentic RAG：模型工厂。

本仓库沿用上游的关键设定「单一共享骨干」（Planner 与所有 Executor 共用同一份权重），
因此这里所有 Agent 都走同一个 ChatOpenAI 实例，仅按角色绑定不同的 temperature
（对应 base_agent_tag.py 中每个 Agent 的 default_temperature）。

额外提供 ScriptedChatModel：不需要真实 LLM 就能把整张图跑通，
用于离线验证图拓扑、条件分支与指标聚合。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import RagConfig


def text_of(message_or_response: Any) -> str:
    """兼容 content 为 str 或 content-block 列表两种返回形态。"""
    content = getattr(message_or_response, "content", message_or_response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


class ScriptedChatModel(BaseChatModel):
    """按 prompt 特征返回合规标签格式的定式输出，供离线联调使用。

    route: linear | qds | qdp —— 控制 PlanningAgent 选择哪条工作流分支。
    """

    route: str = "qds"
    model_name: str = "jade-scripted-mock"

    @property
    def _llm_type(self) -> str:
        return "jade-scripted-mock"

    def _script(self, prompt: str) -> str:
        low = prompt.lower()

        # 1) PlanningAgent：同时出现 "query decomposition" 与 "available tools"
        if "query decomposition" in low and "available tools" in low:
            if self.route == "linear":
                return "<workflow>QR,R,DS,AG</workflow>"
            if self.route == "qdp":
                return "<workflow>QDP</workflow>"
            return "<workflow>QDS</workflow>"

        # 2) SubPlanningAgent：只提供 R / DS / AG
        if "plan a workflow using the provided" in low:
            return "<workflow>R,DS,AG</workflow>"

        # 3) 问题分解（QDS / QDP）
        if "serial decomposition" in low or "decomposing complex multi-entity" in low:
            return (
                "<q1>mock sub-question one</q1>\n"
                "<q2>mock sub-question two</q2>"
            )

        # 4) EvidenceJudgeAgent（检索循环的 judge；必须在 QueryRewriteAgent 之前判断）
        if "research strategist" in low and "enough evidence" in low:
            match = re.search(r"retrieval round (\d+)", low)
            if match and int(match.group(1)) >= 2:
                return "<decision>sufficient</decision>"
            return (
                "<decision>insufficient</decision>\n"
                "<missing>mock missing evidence</missing>\n"
                "<why>mock diagnosis: semantic search only returns passing mentions of the entity</why>\n"
                "<keywords_must>mock, exact, term</keywords_must>\n"
                "<keywords_any>mock synonym, mock alternative</keywords_any>"
            )

        # 5) DocumentSelectionAgent（必须早于 QueryRewriteAgent：
        #    DS/AG 的 prompt 里会拼进检索到的文档，文档文本可能含 "rewritten"）
        if "candidate documents" in low:
            return "<id>Document0,Document1</id>"

        # 5) AnswerSummarizationAgent（必须在 AnswerGenerationAgent 之前判断）
        if "final answer to the original question" in low:
            return "<answer>mock final answer</answer>"

        # 6) 循环内重写（build_retry_query_messages，含 "Queries already tried"）
        if "queries already tried" in low:
            return "<query>mock refined query</query>"

        # 7) QueryRewriteAgent（无 / 有 context 两版 prompt 都以 "skilled at ..." 开头）
        if "skilled at" in low:
            return "<query>mock rewritten query</query>"

        # 7) AnswerGenerationAgent
        if "brief and accurate answer" in low:
            return "<answer>mock answer</answer>"

        return "<answer>mock answer</answer>"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(text_of(m) for m in messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._script(prompt)))]
        )


def build_llm(cfg: RagConfig) -> BaseChatModel:
    """构建共享骨干模型；cfg.mock=True 时返回离线定式模型。"""
    if cfg.mock:
        return ScriptedChatModel(route=cfg.mock_route, model_name=cfg.model)

    from langchain_openai import ChatOpenAI

    # temperature 不在此处固定，交给 Runtime 按 Agent 角色 bind，
    # 以复现 base_agent_tag.py 中不同 Agent 使用不同温度的设定。
    # extra_body 用于把 vLLM 专有参数（如关闭思考模式）透传到请求体。
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        max_tokens=cfg.max_tokens,
        timeout=cfg.retrieval_timeout * 10,
        temperature=None,
        extra_body=cfg.build_extra_body() or None,
    )


__all__ = ["build_llm", "ScriptedChatModel", "text_of", "Sequence"]
