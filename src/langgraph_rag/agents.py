"""LangGraph Agentic RAG：Agent prompt 模板与严格输出解析。

与 qa_manager/base_agent_tag.py 一一对应：
- build_*_messages  <=> 各 Agent 的 _build_messages / _build_workflow_messages
- parse_*           <=> 各 Agent 的 parse_response（返回 {"<字段>": ..., "is_valid": bool}）
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

ALLOWED_MODULES = {"QR", "QDP", "QDS", "R", "DS", "AG"}
MAX_ANSWER_WORDS = 40

EXAMPLE_PROMPT = """
- Example:
Question: When did the simpsons first air on television?
<answer>December 17, 1989</answer>

Question: When did the lightning thief book come out?
<answer>2005</answer>

Question: Who said i'm late i'm late for a very important date?
<answer>The White Rabbit</answer>

Question: Where does the short happy life of francis macomber take place?
<answer>Africa</answer>

Question: What was the fourth expansion pack for sims 2?
<answer>Pets</answer>

Question: Voice of the snake in the jungle book?
<answer>The Jungle Book (2016 film)</answer>

Question: How many seasons are there of star wars the clone wars?
<answer>6</answer>

Question: Which us president appears as a character in the play annie?
<answer>Franklin D. Roosevelt</answer>

Question: Are Calochone and Adlumia both plants?
<answer>yes</answer>

Question: Yukio Mishima and Roberto Bolaño, are Chilean?
<answer>no</answer>
"""


# --------------------------------------------------------------------------------------
# 规则校验（is_workflow_valid）
# --------------------------------------------------------------------------------------
def is_workflow_valid(workflow_str: str) -> bool:
    if not workflow_str:
        return False

    modules = [m.strip() for m in workflow_str.split(",") if m.strip()]

    # 不能重复
    if len(modules) != len(set(modules)):
        return False
    # 基本合法性
    if any(m not in ALLOWED_MODULES for m in modules):
        return False
    # QDP / QDS 只能单独出现
    if "QDP" in modules or "QDS" in modules:
        return len(modules) == 1
    # 最后必须是 AG
    if modules[-1] != "AG":
        return False
    # 顺序约束：QR -> R -> DS -> AG
    order_map = {"QR": 0, "R": 1, "DS": 2, "AG": 3}
    if "QR" in modules and modules[0] != "QR":
        return False
    order_indices = [order_map[m] for m in modules]
    if order_indices != sorted(order_indices):
        return False
    return True


# --------------------------------------------------------------------------------------
# PlanningAgent / SubPlanningAgent
# --------------------------------------------------------------------------------------
def build_planning_messages(query: str, enable_qdp: bool = True) -> List[dict]:
    """整条问句的 workflow 规划（可选 QDS / QDP / 线性链）。"""
    if enable_qdp:
        tools = (
            "  - Query Rewriter (QR): Input: question → Output: rewritten question that is more concise, clearer, and accurate.\n"
            "  - Query Decomposition Serial (QDS): Input: question → Output: dependent sub-questions where later ones depend on earlier ones.\n"
            "  - Query Decomposition Parallel (QDP): Input: question → Output: several sub-questions can be searched independently.\n"
            "  - Retrieval (R): Input: question → Output: relevant candidate documents.\n"
            "  - Document Selector (DS): Input: question + candidate documents → Output: subset of documents helpful for answering.\n"
            "  - Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"
        )
        decompose_rules = (
            "   - If sub-questions have dependencies and must be answered in sequence, use QDS or QDP ONLY.\n"
        )
        only_rule = (
            "   - If you choose QDS or QDP, DO NOT add any other tools/agents.\n"
            "   - The workflow must ONLY contain QDS or QDP in those cases.\n\n"
        )
    else:
        tools = (
            "  - Query Rewriter (QR): Input: question → Output: rewritten question that is more concise, clearer, and accurate.\n"
            "  - Query Decomposition Serial (QDS): Input: question → Output: dependent sub-questions where later ones depend on earlier ones.\n"
            "  - Retrieval (R): Input: question → Output: relevant candidate documents.\n"
            "  - Document Selector (DS): Input: question + candidate documents → Output: subset of documents helpful for answering.\n"
            "  - Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"
        )
        decompose_rules = (
            "   - If sub-questions have dependencies and must be answered in sequence, use QDS ONLY.\n"
        )
        only_rule = (
            "   - If you choose QDS, DO NOT add any other tools/agents.\n"
            "   - The workflow must ONLY contain QDS in those cases.\n\n"
        )

    system = (
        "You are a helpful assistant to plan a Workflow for the given question using the available tools/agents.\n"
        "Your task: Output ONLY the correct workflow inside <workflow>...</workflow> tags following the rules below.\n\n"
        "Available Tools/Agents:\n"
        f"{tools}"
        "Rules for tool selection:\n"
        "1. When the question needs to be broken down into sub-questions:\n"
        f"{decompose_rules}"
        "2. When the question can be answered directly without decomposition:\n"
        "   - Build workflow from QR, R, DS, AG.\n"
        "   - If DS is in workflow, R must appear before DS.\n"
        "   - The last module must always be AG.\n"
        "3. IMPORTANT:\n"
        f"{only_rule}"
        "Example Workflows:\n"
        'Question: "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?"\n'
        "<workflow>QDS</workflow>\n"
        'Question: "Which African country with the fastest economic growth since 1990 had the main export products in 2023?"\n'
        "<workflow>QDS</workflow>\n"
        'Question: "Which team is the 2025 NBA champion?"\n'
        "<workflow>QR,R,AG</workflow>\n\n"
        "Now output in the above format:\n"
    )

    return [
        {"role": "system", "content": system},
        {
            "role": "assistant",
            "content": "Ok, you can provide the Question and I will give the appropriate Workflow inside <workflow>...</workflow> tags.",
        },
        {
            "role": "user",
            "content": f'Please give the appropriate Workflow in the format requested above.\nQuestion: "{query}"\nWorkflow: ',
        },
    ]


def build_sub_planning_messages(query: str) -> List[dict]:
    """子问题的 workflow 规划（只允许 R / DS / AG）。"""
    system = (
        "You are a helpful assistant to plan a Workflow using the provided tools/agents "
        "based on the given question.\n"
        "Your task: Output ONLY the workflow inside <workflow>...</workflow> tags.\n\n"
        "Available tools/agents:\n"
        "Retrieval (R): Input: question → Output: relevant candidate documents.\n"
        "Document Selector (DS): Input: question + candidate documents → Output: documents helpful for answering.\n"
        "Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"
        "Rules:\n"
        "1. If DS is in the workflow, R must appear before DS.\n"
        "2. The final module must always be AG.\n\n"
        "Examples:\n"
        "Question: 'Which city is the capital of Australia.'\n"
        "<workflow>AG</workflow>\n\n"
        "Question: 'Which team is the 2025 NBA champion?'\n"
        "<workflow>R,AG</workflow>\n\n"
        "Now I will give you a question. Just output the appropriate Workflow inside <workflow>...</workflow> tags, "
        "and do not include anything else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Ok, you can provide the question and I will give the appropriate Workflow."},
        {
            "role": "user",
            "content": f'Please give the appropriate Workflow in the required format.\nQuestion: "{query}"\nWorkflow: ',
        },
    ]


def parse_workflow(text: str) -> Dict[str, object]:
    """严格解析 <workflow>...</workflow>。

    与原实现一致：标签外有内容 / 缺标签 => is_valid=False。
    额外兜底：模型漏写开标签或残留闭标签时，仍尽量抽出可执行文本，
    但依然判为非法（保持 -2.0 penalty 的语义）。
    """
    text = (text or "").strip()
    strict = re.fullmatch(r"\s*<workflow>\s*(.*?)\s*</workflow>\s*", text, flags=re.I | re.S)
    if strict:
        content = strict.group(1).strip()
        return {"workflow": content, "is_valid": is_workflow_valid(content)}

    loose = re.search(r"<workflow>(.*?)</workflow>", text, flags=re.I | re.S)
    if loose:
        return {"workflow": loose.group(1).strip(), "is_valid": False}
    return {"workflow": re.sub(r"</?workflow>", "", text, flags=re.I).strip(), "is_valid": False}


# --------------------------------------------------------------------------------------
# QueryRewriteAgent
# --------------------------------------------------------------------------------------
def build_query_rewrite_messages(query: str, context: Optional[Dict[str, Any]] = None) -> List[dict]:
    if not context:
        return _qr_plain(query)
    return _qr_with_context(query, context)


def _qr_plain(query: str) -> List[dict]:
    system = (
        "You are a professional assistant skilled at rewriting slightly redundant or overly wordy factual questions "
        "into a single, concise, and searchable query. Keep all essential names, dates, and terms. "
        "Do not add explanations or unrelated details.\n"
        "You must output ONLY the rewritten query inside <query>...</query> tags, without any extra text outside the tags."
    )
    user = (
        f"Original question is: {query}\n"
        "Rewrite the original question into a short, clear query containing only the key searchable information. "
        "Preserve important names, places, dates, and events.\n\n"
        "Follow the format shown in the examples:\n\n"
        "Example 1:\n"
        "Original question: 'Who was the Prime Minister of the United Kingdom when the Brexit referendum took place in 2016?'\n"
        "Rewritten query: <query>UK Prime Minister during 2016 Brexit referendum</query>\n\n"
        "Example 2:\n"
        "Original question: 'The novel Beloved by Toni Morrison won which major literary award in 1988?'\n"
        "Rewritten query: <query>Major literary award won by Beloved in 1988</query>\n\n"
        "Example 3:\n"
        "Original question: 'In what year did the Voyager 1 spacecraft first enter interstellar space?'\n"
        "Rewritten query: <query>Year Voyager 1 entered interstellar space</query>\n\n"
        "Example 4:\n"
        "Original question: 'Marie Curie was awarded Nobel Prizes in which scientific fields?'\n"
        "Rewritten query: <query>Scientific fields of Marie Curie’s Nobel Prizes</query>\n\n"
        "Example 5:\n"
        "Original question: 'The Great Wall of China spans across how many provinces?'\n"
        "Rewritten query: <query>Number of provinces spanned by Great Wall of China</query>\n\n"
        f"Original question is: {query}\n"
        "Rewritten query: "
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Okay, I will return the concise rewritten query inside <query>...</query> tags."},
        {"role": "user", "content": user},
    ]


def _qr_with_context(query: str, context: Dict[str, Any]) -> List[dict]:
    sub_queries_info = "\n".join(
        [
            f"Sub-query: {item['query']}\nAnswer: {item.get('answer', '')}"
            for item in context.get("sub-query_docs_a", [])
        ]
    )
    system = (
        "You are a professional assistant skilled at synthesizing context from a main query and related sub-queries "
        "with their answers, then rewriting a follow-up query into a single, concise, and searchable form. "
        "You must ensure the rewritten query captures the essential meaning of the current query while leveraging the given context. "
        "Preserve important names, dates, places, and facts. Do not add explanations or unrelated details.\n"
        "You must output ONLY the rewritten query inside <query>...</query> tags, without any extra text outside the tags."
    )
    user = (
        f"Main original query:\n{context.get('original_query', '')}\n\n"
        f"Related sub-queries & answers:\n{sub_queries_info}\n\n"
        f"Current query to rewrite:\n{query}\n\n"
        "Based on the above, rewrite the current query into a single, clear, and searchable query "
        "that includes only the essential details needed to find an answer efficiently. Avoid unnecessary context or verbose wording.\n\n"
        "Follow the format shown in the examples:\n\n"
        "Example 1:\n"
        "Original question: 'Who was the Prime Minister of the United Kingdom when the Brexit referendum took place in 2016?'\n"
        "Rewritten query: <query>UK Prime Minister during 2016 Brexit referendum</query>\n\n"
        "Example 2:\n"
        "Original question: 'The novel Beloved by Toni Morrison won which major literary award in 1988?'\n"
        "Rewritten query: <query>Major literary award won by Beloved in 1988</query>\n\n"
        "Example 3:\n"
        "Original question: 'T. A. Sarasvati Amma was a scholar born in a country with a population of over how many people?'\n"
        "Rewritten query: <query>Population of T. A. Sarasvati Amma’s birth country</query>\n\n"
        f"Original question is: {query}\n"
        "Rewritten query: "
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Okay, I will rewrite the current query inside <query>...</query> tags."},
        {"role": "user", "content": user},
    ]


def parse_query_rewrite(text: str, init_question: str) -> Dict[str, object]:
    strict = re.fullmatch(r"\s*<query>\s*(.*?)\s*</query>\s*", text or "", flags=re.I | re.S)
    if not strict:
        return {"query_rewrited": (text or "").strip(), "is_valid": False}
    extracted = strict.group(1).strip()
    word_count = len([w for w in extracted.split() if w.strip()])
    is_valid = (extracted != init_question) and (word_count <= 100)
    return {"query_rewrited": extracted, "is_valid": is_valid}


# --------------------------------------------------------------------------------------
# QueryDecompositionAgentParallel / Serial
# --------------------------------------------------------------------------------------
def build_decomposition_messages(query: str, serial: bool = True) -> List[dict]:
    if serial:
        system = (
            'You are a professional assistant skilled at decomposing complex questions into a minimal sequence '
            'of logically dependent, independently searchable sub-questions (serial decomposition). '
            'Each sub-question must:\n'
            '- Be self-contained and specific\n'
            '- Be suitable for direct information retrieval from search engines or structured databases\n'
            '- Be strictly necessary to answer the original question\n'
            '- Form a logical chain, where later sub-questions depend on earlier sub-questions\n'
            'You must keep the number of sub-questions as low as possible, never exceed 4 in total, and avoid redundancy.\n'
            'You must output each sub-question inside numbered tags <q1>...</q1>, <q2>...</q2>, etc., starting at q1 in ascending order, '
            'with no gaps, no repeats, and no extra text outside tags.'
        )
        assistant = 'Understood. I will return each sub-question wrapped in its tag, in strict sequence.'
        user = (
            f'Original question is: {query}\n'
            'Now decompose the original question into a logically ordered sequence of dependent sub-questions. '
            'Output one sub-question per line, each inside the correct numbered tag.\n\n'
            'Example:\n'
            "Original question: 'Which African country with the fastest economic growth since 1990 had the main export products in 2023?'\n"
            "<q1>Which African country had the fastest economic growth since 1990?</q1>\n"
            "<q2>What were the main export products of that country in 2023?</q2>\n\n"
            f'Original question: {query}\n'
            'Sub-questions:\n'
        )
        return [
            {"role": "system", "content": system},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": user},
        ]

    system = (
        "You are a professional assistant skilled at decomposing complex multi-entity or multi-location "
        "questions into multiple independent and searchable sub-questions. "
        "Each sub-question should be specific, logically complete, and avoid duplication.\n"
        "You must output each sub-question on its own line, wrapped in numbered tags <q1>...</q1>, <q2>...</q2>, etc., "
        "starting from <q1> in ascending order with no gaps, no repeats, and no extra text outside the tags."
    )
    user = (
        f"Original question is: '{query}'.\n"
        "Break down this question into the minimum number of specific, logically complete, "
        "and independently searchable sub-questions needed to fully understand and answer "
        "the original question. Do not generate more than 4 sub-questions.\n"
        "Each sub-question should be on a separate line and tagged as follows:\n"
        "<q1>first sub-question</q1>\n"
        "<q2>second sub-question</q2>\n"
        "... etc.\n\n"
        "Example:\n"
        "Original question: 'What are the main exports of Germany and Japan in 2023?'\n"
        "<q1>What were the main exports of Germany in 2023?</q1>\n"
        "<q2>What were the main exports of Japan in 2023?</q2>\n\n"
        f"Original question: '{query}'\n"
        "Sub-questions:\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Okay, I will return each sub-question tagged individually in sequence."},
        {"role": "user", "content": user},
    ]


def parse_subquestions(text: str, init_question: str, max_sub_questions: int = 7) -> Dict[str, object]:
    """严格解析 <q1>..<qN>，标签必须连续递增且标签外无内容。"""
    raw = (text or "").strip()
    invalid = {"subquestions": [init_question], "is_valid": False}

    structure = re.compile(r"\s*(<q\d+>.*?</q\d+>\s*)+", flags=re.S | re.I)
    if not structure.fullmatch(raw):
        return invalid

    tags = re.findall(r"<q(\d+)>(.*?)</q\1>", raw, flags=re.S | re.I)
    if not tags:
        return invalid

    expected = list(range(1, len(tags) + 1))
    actual = [int(num) for num, _ in tags]
    if actual != expected:
        return invalid

    subquestions = [t.strip() for _, t in tags]
    count = len(subquestions)
    is_valid = count <= max_sub_questions
    if len(set(subquestions)) != count or any(s == init_question.strip() for s in subquestions):
        is_valid = False

    return {"subquestions": subquestions, "is_valid": is_valid}


# --------------------------------------------------------------------------------------
# DocumentSelectionAgent
# --------------------------------------------------------------------------------------
def build_doc_selection_messages(query: str, documents: List[str]) -> List[dict]:
    doc_content = "".join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
    max_id = len(documents) - 1

    return [
        {
            "role": "system",
            "content": (
                f"You are a helpful, respectful and honest assistant. "
                f"Your task is to output the IDs of the candidate Documents (0, 1, 2,..., {max_id}) "
                f"which are helpful in answering the Question. "
                f"You must output ONLY inside <id>...</id> tags, as comma-separated values like "
                f"<id>Document0,Document2,Document4</id>. There must be no text outside the tags."
            ),
        },
        {"role": "assistant", "content": "Okay, I will provide the IDs of candidate Documents inside <id>...</id> tags."},
        {"role": "user", "content": f"Question is: {query}\n\n{doc_content}"},
        {"role": "assistant", "content": "OK, I received the Question and the candidate Documents."},
        {
            "role": "user",
            "content": (
                f"Now, output the IDs of the candidate Documents (0,1,2,...,{max_id}) "
                "which are helpful in answering the Question. Format them as comma-separated values inside <id>...</id> tags.\n"
                "Example: <id>Document0,Document2,Document4</id>\n"
                "Do not output any other text outside the tags."
            ),
        },
    ]


def parse_doc_ids(text: str, total_docs: int) -> Dict[str, object]:
    valid_ids = set(range(total_docs))
    all_docs = sorted(valid_ids)

    strict = re.fullmatch(r"\s*<id>\s*(.*?)\s*</id>\s*", text or "", flags=re.S | re.I)
    if not strict:
        return {"docs_ids": all_docs, "is_valid": False}

    inside = strict.group(1).strip()
    if not re.fullmatch(r"(Document\d+(?:,Document\d+)*)?", inside):
        return {"docs_ids": all_docs, "is_valid": False}

    ids_list = [int(m) for m in re.findall(r"Document(\d+)", inside)]
    if not ids_list:
        return {"docs_ids": [], "is_valid": True}
    if not all(i in valid_ids for i in ids_list):
        return {"docs_ids": sorted(i for i in ids_list if i in valid_ids), "is_valid": False}

    ids_set = sorted(set(ids_list))
    if len(ids_list) != len(ids_set):
        return {"docs_ids": ids_set, "is_valid": False}
    return {"docs_ids": ids_set, "is_valid": True}


# --------------------------------------------------------------------------------------
# AnswerGenerationAgent / AnswerSummarizationAgent
# --------------------------------------------------------------------------------------
# P2.3 答案格式约束：2wiki 案例显示 comparison 题输出对照枚举（"A: 1992; B: 2002"）
# 或答非所问（yes/no 题答国家名），judge 判错——检索全命中时格式是主要失分点。
_YESNO_RE = re.compile(
    r"^(are|is|was|were|do|does|did|can|could|should|would|has|have|had)\b", re.I
)


def answer_format_hint(query: str) -> str:
    """按题型给出输出格式约束（轻量规则，无外部依赖）。"""
    q = str(query or "").strip()
    if _YESNO_RE.match(q):
        return " This is a yes/no question: answer with exactly one word, Yes or No."
    if re.match(r"^(which|who|whom)\b", q, re.I) and re.search(r"\bor\b", q, re.I):
        return (
            " This is a choice question: answer with ONLY the exact correct option itself"
            " (an entity or name), without dates, comparison or any extra words."
        )
    return (
        " Answer with the shortest exact span (an entity name, a date or a number),"
        " not a full sentence."
    )


def build_answer_messages(
    query: str, documents: Optional[List[str]] = None, format_query: str = ""
) -> List[dict]:
    format_instruction = (
        "Please answer the Question strictly following the format shown in the examples above. "
        "You must output ONLY inside <answer>...</answer> and nothing outside these tags."
        + answer_format_hint(format_query or query)
    )

    # 格式约束同时注入 system 与 user 末端：27B 骨干对远离生成位置的
    # system 指令遵循不足（'Which film ... A or B?' 仍答日期枚举），
    # user 末端最贴近生成位置、遵循率最高。
    hint = answer_format_hint(format_query or query)
    if documents:
        doc_content = "".join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
        system = (
            f"You are a helpful, respectful and honest assistant. "
            f"Your task is to provide a brief and accurate answer to the Question based on the given Documents. "
            f'If you do not know the answer from the Documents, say "I don\'t know" and do not make up information.\n'
            f"{EXAMPLE_PROMPT}\n{format_instruction}"
        )
        user = (
            f"Question: {query}\n\n{doc_content}"
            f"Now, give the brief and accurate answer inside <answer>...</answer> tags.{hint}"
        )
    else:
        system = (
            f"You are a helpful, respectful and honest assistant. "
            f"Your task is to provide a brief and accurate answer to the Question using only the information you know. "
            f'If you do not know the answer, say "I don\'t know" and do not make up information.\n'
            f"{EXAMPLE_PROMPT}\n{format_instruction}"
        )
        user = (
            f"Question: {query}\n\n"
            f"Now, give the brief and accurate answer inside <answer>...</answer> tags.{hint}"
            "You must not answer \"I don't know.\" Always provide the answer you think is most correct."
        )

    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Okay, I will answer the Question inside <answer>...</answer> tags."},
        {"role": "user", "content": user},
    ]


def build_answer_summarization_messages(
    query: str, sub_queries: List[str], sub_answers: List[str]
) -> List[dict]:
    format_instruction = (
        "Please answer the Original Question strictly following the format shown in the examples above. "
        "You must output ONLY inside <answer>...</answer> and nothing outside these tags."
        + answer_format_hint(query)
    )
    observation = "".join(
        [
            f"Sub-question {i + 1}: {sq}\nAnswer: {sa}\n\n"
            for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers))
        ]
    )
    system = (
        f"You are a helpful, respectful and honest assistant. "
        f"Your task is to predict the final answer to the Original Question based on the answers to its decomposed sub-questions. "
        f"If you are not sure about the final answer, do not make up information.\n"
        f"{EXAMPLE_PROMPT}\n{format_instruction}"
    )
    user = (
        f"Original Question: {query}\n\n"
        f"{observation}"
        f"Now, based on the above sub-questions and their answers, "
        f"answer the Original Question: {query} inside <answer>...</answer> tags. "
        f"{answer_format_hint(query)} "
        'You must not answer "I don\'t know." Always provide the answer you think is most correct.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Okay, I will provide the final answer inside <answer>...</answer> tags."},
        {"role": "user", "content": user},
    ]


def parse_answer(text: str) -> Dict[str, object]:
    """严格解析 <answer>...</answer>，并要求不超过 40 个词。"""
    if not isinstance(text, str) or not text.strip():
        return {"answer": text, "is_valid": False}

    strict = re.fullmatch(r"\s*<answer>\s*(.*?)\s*</answer>\s*", text, flags=re.S | re.I)
    if not strict:
        return {"answer": text.strip(), "is_valid": False}

    answer_text = strict.group(1).strip()
    if len(re.findall(r"\w+", answer_text)) > MAX_ANSWER_WORDS:
        return {"answer": text.strip(), "is_valid": False}
    return {"answer": answer_text, "is_valid": True}


# --------------------------------------------------------------------------------------
# EvidenceJudgeAgent（INTERACT-RAG 的 Adaptive-Reasoner）与循环内的 query 重写
#
# 这是「检索→判断→不足→重写→再检索」循环里新增的角色，原仓库没有：
# - PlanningAgent 不感知循环：plan 阶段只输出 workflow（形状），由运行期的 judge
#   在真正看到检索结果后决定「够不够、要不要再来一轮」。
# --------------------------------------------------------------------------------------
_JUDGE_TAG_RE = re.compile(
    r"<(decision|missing|why|keywords_must|keywords_any|keywords|entity|exclude)>(.*?)</\1>",
    flags=re.I | re.S,
)


def build_evidence_check_messages(
    query: str, documents: List[str], round_no: int, max_rounds: int
) -> List[dict]:
    """判断当前累积文档是否足以回答 query；不足时**先诊断失败原因**再给出下一轮策略。

    对齐 INTERACT-RAG §3.2 的 Adaptive-Reasoner：Reflect 不是换个说法重搜，而是
    先说明「为什么没检索到」，再据此切换检索原语（精确词 / 实体锚定 / 排除文档）。
    """
    doc_content = "".join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
    system = (
        "You are an expert research strategist. Your task is to evaluate whether the retrieved "
        "Documents already contain enough evidence to answer the Question, and to decide whether "
        "one more retrieval round is needed. Do not answer the question.\n"
        "You must output ONLY the tags listed below, with no text outside them:\n"
        "- If the evidence is sufficient: <decision>sufficient</decision>\n"
        "- If the evidence is insufficient, output:\n"
        "    <decision>insufficient</decision>\n"
        "    <missing>the specific information that is still missing</missing>\n"
        "    <why>a brief diagnosis of WHY the current retrieval failed (do not just restate the question)</why>\n"
        "  Diagnose the failure mode first, then pick the strategy it calls for:\n"
        "    * The answer is a concrete fact (a date, number, or name), but the documents only "
        "mention the entity in passing -> put the fact word itself in <keywords_must> "
        "(e.g. \"born\", \"founded\", \"released\", \"died\").\n"
        "    * Results are dominated by a same-named entity or unrelated news -> anchor the search "
        "on the specific entity so that distractors are filtered out.\n"
        "    * The needed evidence is phrased differently from the query -> put literal terms that "
        "must appear in <keywords_must>, instead of paraphrasing the question again.\n"
        "When insufficient, you MAY add optional tags to guide the next search:\n"
        "- <keywords_must>1 to 3 concrete terms that MUST appear literally in the target text "
        "(e.g. \"born\"); separate multiple terms with commas</keywords_must>\n"
        "- <keywords_any>optional alternative phrasings or synonyms of the evidence, separated by "
        "commas; these are only a weak hint and must NOT contain a long list of unrelated synonyms</keywords_any>\n"
        "- <entity>one specific entity the next search should be anchored on</entity>\n"
        "- <exclude>comma separated Document ids that are irrelevant and must be dropped, e.g. 1,3</exclude>"
    )
    assistant = (
        "Understood. I will output only <decision>, <missing>, <why>, <keywords_must>, "
        "<keywords_any>, <entity>, <exclude> tags."
    )
    user = (
        f"Question: {query}\n\n{doc_content}"
        f"This is retrieval round {round_no} of at most {max_rounds}. "
        "Decide whether the Documents above are sufficient to answer the Question."
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": user},
    ]


def parse_evidence_check(text: str) -> Dict[str, object]:
    """解析 judge 输出。

    返回 {"sufficient", "missing", "why", "exclude", "plan", "is_valid"}：
    - why   ：Reflect 的失败诊断（可选字段，不参与合法性判定，仅作审计/重写上下文）
    - plan  ：要传给引擎的动作原语；judge 给的 keywords 走「必须出现」(and) 语义
    - is_valid=False 时（格式非法）保守当「不足」，继续重写再检索，但会被 max_rounds 截断
    """
    raw = (text or "").strip()
    tags = {key.lower(): value.strip() for key, value in _JUDGE_TAG_RE.findall(raw)}
    outside = _JUDGE_TAG_RE.sub("", raw).strip()
    decision = str(tags.get("decision", "")).lower()

    is_valid = (not outside) and decision in ("sufficient", "insufficient")
    if is_valid and decision == "insufficient":
        is_valid = bool(tags.get("missing"))

    if not is_valid:
        return {
            "sufficient": False,
            "missing": "",
            "why": "",
            "exclude": [],
            "plan": {},
            "is_valid": False,
        }

    def _terms(value: str) -> List[str]:
        """按逗号/分号切成 term（词组），例如 'born, released' -> ['born', 'released']。"""
        return [t.strip() for t in re.split(r"[,;]", value or "") if t.strip()]

    must = _terms(tags.get("keywords_must", ""))
    any_terms = _terms(tags.get("keywords_any", ""))
    legacy = _terms(tags.get("keywords", ""))
    if legacy:
        # 兼容旧字段 <keywords>：≤3 个 term 当作「必须出现」，长同义词列表当作 any
        (must if len(legacy) <= 3 else any_terms).extend(legacy)

    # 兜底：judge 常把「话题词列表」塞进 must（实测 4~5 个），逐个 AND 必然空集 ->
    # 回退 OR 等于空转。超过 3 个就降级为 any（弱提示），保证 AND 只在真正的
    # 「1~3 个决定性事实词」时才生效（对齐论文 exact_query_keywords 的用法）。
    if len(must) > 3:
        any_terms.extend(must)
        must = []

    entity = tags.get("entity", "")
    exclude = [int(m) for m in re.findall(r"\d+", tags.get("exclude", ""))]

    plan: Dict[str, object] = {}
    if must:
        # 「必须出现」：judge 点名的事实词（如 born）不能只是"任一命中"
        plan["keywords_must"] = must
        plan["weights"] = (0.4, 0.6)
    if any_terms:
        plan["keywords_any"] = any_terms
        plan.setdefault("weights", (0.6, 0.4))  # 只有同义词时别让词法主导
    if entity:
        plan["entity"] = [entity]

    return {
        "sufficient": decision == "sufficient",
        "missing": tags.get("missing", ""),
        "why": tags.get("why", ""),
        "exclude": exclude,
        "plan": plan,
        "is_valid": True,
    }


def build_retry_query_messages(
    query: str, tried_queries: List[str], missing_info: str, failure_reason: str = ""
) -> List[dict]:
    """循环内重写：结合「已试过的 query + 还缺什么 + 为什么失败」生成新的检索 query。

    把 judge 的失败诊断（failure_reason）显式喂给重写，避免退化成盲目改写同义问法。
    """
    tried = "\n".join(f"- {q}" for q in tried_queries) or "- (none)"
    system = (
        "You are a professional assistant skilled at refining a search query after an unsuccessful "
        "retrieval attempt. Given the original question, the queries already tried, the information "
        "that is still missing, and a diagnosis of why the previous attempt failed, produce a NEW query "
        "that is more likely to retrieve the missing evidence. Prefer precise and distinctive terms "
        "(specific names, dates, numbers, or literal fact words such as \"born\") over loose paraphrases, "
        "and never repeat a query that has already been tried.\n"
        "You must output ONLY the refined query inside <query>...</query> tags, without any extra text."
    )
    assistant = "Okay, I will return the refined query inside <query>...</query> tags."
    user = (
        f"Original question: {query}\n"
        f"Queries already tried:\n{tried}\n\n"
        f"Information still missing: {missing_info or 'the evidence needed to answer the question'}\n"
        f"Why the previous attempt failed: {failure_reason or '(not diagnosed)'}\n\n"
        "Refined query: "
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": user},
    ]
