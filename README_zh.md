# LangGraph Agentic RAG

[English README](README.md)

用 **LangGraph** 实现的 agentic RAG 工作流，融合 **INTERACT-RAG 风格的运行期检索
循环**、**GAM-RAG 风格的免训练自进化记忆**，检索侧扩展了多面检索、实体锚定与
模型级重排。单一共享 LLM 骨干扮演全部 Agent 角色；图按问题动态编排工作流
（`QR → R → DS → AG`、`QDS`、`QDP`……），检索经验跨轮次累积，系统随评测轮数
**又准又快**，全程无需任何梯度更新。

## 亮点

- **9 个 Agent、一个骨干**——Planning、SubPlanning、QueryRewrite (QR)、
  QueryDecomposition 并行/串行 (QDP/QDS)、DocumentSelection (DS)、
  AnswerGeneration (AG)、AnswerSummarization、EvidenceJudge，各自独立的
  prompt、温度与严格输出解析器。
- **动态工作流**——PlanningAgent 按题选择链形状；非法 plan 回退默认工作流并
  记惩罚（与奖励口径兼容）。
- **运行期检索循环**——EvidenceJudge 在运行期判断证据是否充足；不足则携带
  「还缺什么」上下文重写 query 再检索（轮数上限 `RAG_MAX_ROUNDS`）。
- **自进化记忆（免训练）**——段落级 Kalman 记忆（吸引有用段落、推离被判无关
  段落、持久排除、记忆激活注入）+ 策略级 `semantic_key → judge_plan`
  warm-start（四重可信门：cos / π / 支持度 / ≥3 次独立成功，超容按支持度淘汰）。
  反馈可来自金标（`gold`，评测口径）或 Agent 自身的 judge/DS 判定
  （`agent`，生产口径）。
- **语料交互引擎**——向量 + BM25 + 实体加权的混合检索，带记忆门控，
  全部评测使用。
- **模型级重排（可选）**——engine 候选池 30 篇由 cross-encoder 重排到 `top_k`，
  可切 Qwen3-Reranker-4B（默认）或 NeoHorse-Jev-4B 决策模型（单次 forward 重排
  整池，另有 tournament 图排序）。
- **可复现评测套件**——QASPER（论文段落单跳）与 HotpotQA / 2WikiMultihopQA /
  MuSiQue（多跳，GAM-RAG *k-turn memorization* 协议，三臂：`off` / `agent` / `gold`）。

## 工作流路由

PlanningAgent 只看到问题本身，在 `<workflow>...</workflow>` 里输出一个标签，链形状
随后固定不变。可路由的形态有 3 类、实际有效标签 7 个：

| 工作流 | 去向 | 例子 | Planner 的先验判断 |
|---|---|---|---|
| `QDS` | decompose（串行子问题） | 「1990 年以来经济增长最快的非洲国家，2023 年的主要出口产品是什么？」 | 子问题有依赖，后者要用前者的答案 |
| `QDP` | decompose（并行子问题） | 「2024 年 GDP 最高的城市：新加坡、休斯顿还是北京？」 | 子问题互相独立，可同时检索 |
| `AG` | linear | 「3.5 的平方是多少？」或闲聊 | 参数化知识就能答，不需要语料 |
| `R,AG` | linear | 「2025 年 NBA 总冠军是哪支球队？」 | 问句已适合检索，且预期命中唯一 |
| `R,DS,AG` | linear | 「sims 2 的第四部资料片是什么？」 | 问句清楚，但语料里同实体段落多，候选需要筛 |
| `QR,R,AG` | linear | 「The Lightning Thief 这本书什么时候出版的？」 | 问句需要重写（书名非正式），但答案唯一、无需筛 |
| `QR,R,DS,AG` | linear | 「"我迟了我迟了，我有一个重要约会"是谁说的？」 | 问句口语化或有歧义，且预期候选杂乱 |

合法集合是所有保持 QR→R→DS→AG 顺序、以 AG 结尾的 `{QR, R, DS, AG}` 子集，
共 8 个。`QR,AG`、`DS,AG`、`QR,DS,AG` 能过顺序校验，但 prompt 规则把它们排除了
（DS 前面必须有 R），所以 planner 实际是在上面 7 个里选。非法标签回退
`RAG_DEFAULT_WORKFLOW`（默认 `QDS`）并记格式惩罚。子问题用的是更短的菜单，
SubPlanningAgent 只允许 `R`、`DS`、`AG`。

Planner 的先验判断，按权重顺序：

1. **要不要用语料**——闲聊、算术、模型已知的事实走 `AG`，其余至少保留 `R`
2. **问句是否适合直接检索**——口语引用、非正式书名、依赖上下文的表述需要 QR
   先规范化
3. **预期候选有多杂**——热门实体（"sims 2"）、宽泛问题（"总结这篇论文"）、
   多来源话题意味着大量同实体段落，`DS` 才划得来；窄事实（"2025 冠军"）不需要
4. **子问题是否互相依赖**——依赖走 `QDS`，独立走 `QDP`，都不沾就保持线性

静态 plan 判错有两层缓冲：AG 会把全部候选文档收进 prompt 自己挑
（`agents.py`），运行期 judge 循环在证据不足时带缺失信息重写再检索一轮。

## 核心结果


| 设置（40–60 题、5 轮、temp=0、Qwen 27B judge） | 结果 |
|---|---|
| QASPER 单跳，`agent` 反馈 | Judge-Acc **+11pp**（vs 0-turn） |
| HotpotQA，`gold` 反馈 | Judge-Acc **0.925 (+3pp)**、F1 **+5pp**、检索轮数 **−16%**，双曲线单调 |
| 2WikiMultihopQA | 中性（基线已高、语料最小）；仅答案格式修复即抬升基线 **+15pp** |
| MuSiQue 2–4 跳，`agent` 反馈 | **−7.5pp 负进化**——长链上自我标注反馈不可靠（warm-start 污染，消融定位根因） |

**结论**：自进化收益 = f(反馈质量 × 任务难度)。gold 反馈全谱系安全；agent
自我标注在 2-hop 内成立，更长链需关闭策略通道（或接入 GT 级反馈）。

## 各组件的收益

均在 40 题 / 5 轮、temp=0 下测得（`agent` 或 `gold` 反馈）。

| 组件 | 出处 | 实测收益 |
|---|---|---|
| 多面检索 + 实体锚定 | Interact-RAG | 单轮检索的证据命中率 0.47 → 0.53–0.63 |
| 运行期检索循环（EvidenceJudge） | Interact-RAG | QASPER F1 0.25 → 0.48（temp=0，14 题）、证据命中 0.21 → 0.57；代价是检索轮数 1.0 → 2.1——本仓库单项增益最大的改动 |
| 自进化记忆（段落级 + 策略级） | GAM-RAG | `gold` 5 轮：证据命中 **+16.5pp**（0.575 → 0.740）、检索轮数 **−14%**、F1 +3.6pp。`agent`（无金标）10 轮：证据命中 **+11.0pp**、轮数 −12%，命中率单调升到 0.750 |
| 策略记忆 warm-start | GAM-RAG | 第 10 轮时 40/40 题命中策略库（`agent`）；第 5 轮起反超仅文档记忆（0.775 vs 0.750） |
| 模型级重排 | BlitzRank + cross-encoder | MuSiQue 300 道 unseen 题、同一 top-30 候选池：hit@8 **0.9867**（Qwen3-Reranker-4B）vs 0.9600（Jev-4B），mrr@8 0.846 vs 0.714；1000 篇候选池 nDCG@10 **0.82**。可选，默认关 |
| 成本感知奖励 | Dynamic Harness Search | 奖励 = task score − 资源惩罚，再 rescale 成 utility；14 条真实轨迹与上游训练侧逐项比对 **0** 处不一致，judge agent 可直接进 PPO |

## 目录结构

```
.
├── src/langgraph_rag/        # 核心包
│   ├── config.py             # env 驱动配置（见 .env.example）
│   ├── state.py              # RagState / ChainState / TraceStep
│   ├── agents.py             # 9 个 Agent：prompt + 严格解析
│   ├── graph.py              # LangGraph 图拓扑（plan → chain → 检索循环）
│   ├── runtime.py            # Agent 调用、记忆反馈归因
│   ├── engine.py             # 语料交互引擎（向量+BM25+实体+记忆门控）
│   ├── memory.py             # 自进化记忆（Kalman，段落级 + 策略级）
│   ├── vector.py             # embedding 缓存 + 余弦检索
│   ├── rerank.py             # 模型级重排（Qwen3 cross-encoder / Jev 决策模型）
│   ├── jev_decisions.py      # Jev 决策客户端（OpenAI Decisions API 同构）
│   ├── llm.py                # OpenAI 兼容 chat model + 脚本化 mock
│   ├── reward.py             # F1 / EM / judge 指标
│   └── tools.py              # 检索工具接口
├── eval/
│   ├── qasper_eval.py        # 单跳评测（QASPER）
│   └── multihop_eval.py      # 多跳 k-turn memorization（GAM-RAG 协议）
├── scripts/
│   ├── run.py                # CLI 入口：单题 / 批量 / mermaid 图
│   └── walkthrough.py        # 单条 query 逐步演练
├── datasets/                 # 数据自备（见 datasets/README.md）
├── results/                  # 评测输出（gitignored）
├── .env.example              # 全部配置项与默认值
└── requirements.txt
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # 或 pip install -e .
cp .env.example .env                 # 编辑端点/模型后：
set -a; source .env; set +a
```

要求：Python ≥ 3.10、一个 OpenAI 兼容 LLM 端点（如 vLLM），可选 embeddings
端点。mock 模式下完全离线可跑（无需任何服务）。

## 快速开始

```bash
# 离线 mock——验证图拓扑，无需 LLM/检索服务
python scripts/run.py --mock --mock-route qds \
    --question "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?" \
    --gold "mock final answer"

# 真实后端（vLLM）+ 本地语料 + 运行期检索循环 + 记忆
python scripts/run.py --base-url http://127.0.0.1:8000/v1 --model <model> \
    --retrieval-mode engine --corpus corpus.jsonl --max-rounds 3 \
    --question "..." --gold "..." --show-process

# 打印 Mermaid 流程图
python scripts/run.py --mermaid

# 记忆机制离线自测（Kalman 动力学/门控/排除/注入/策略 warm-start/
# defer-commit/持久化/空库回归）
PYTHONPATH=src python -m langgraph_rag.memory
```

## 评测

先按 [`datasets/README.md`](datasets/README.md) 下载并摆放数据集。

```bash
# 单跳（QASPER）：12 篇论文构建语料，评测 24 题
python eval/qasper_eval.py --base-url http://127.0.0.1:8000/v1 --model <model> \
    --embed-url http://127.0.0.1:10200/v1 --papers 12 --questions 24 \
    --workers 4 --scope open --out results/qasper.json

# 多跳，GAM-RAG k-turn memorization，三臂
python eval/multihop_eval.py --dataset hotpotqa --questions 40 --turns 5 \
    --arms off,agent,gold --base-url http://127.0.0.1:8000/v1 --model <model> \
    --workers 3 --temperature 0 --out results/mh_hotpotqa.json
# --dataset: hotpotqa | 2wiki | musique ；--arms: off | agent | gold
```

记忆模式由 `RAG_MEMORY_FEEDBACK`（`off`/`gold`/`agent`）与 `RAG_MEMORY_DB`
（SQLite 持久化，跨轮进化）控制；评测脚本会按臂自动接线。

## 参考文献

本仓库在实现中对齐了以下工作，各论文在代码里的落点见表。

| 论文 | 在本仓库的落点 |
|---|---|
| **JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG** (ICML 2026) | 图拓扑与按题动态编排（`graph.py`）、9 个 Agent 的 prompt 与严格解析（`agents.py`）、指标定义与按 Agent 的 step 聚合（`reward.py`） |
| **Dynamic Harness Search: Building Multi-Agent Systems Per-Query via Prediction** (arXiv 2026) | 奖励口径照它的 cost-aware execution reward 来。本仓库对应 `reward.py` 的 `penalty` + `final_reward` → `whole`/`step_reward`，task score 用 EM/F1，成本项落在格式合法性与 turn/retrieval 次数上。其 per-query harness 预测器未实现 |
| **Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval** (ICLR 2026) | Corpus Interaction Engine 的多面检索、锚定匹配与上下文整形（`engine.py`）；运行期检索循环的 EvidenceJudge 回路（`graph.py`） |
| **GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval** (ICML 2026) | 段落级 Kalman 记忆、持久排除、记忆激活注入与策略记忆 warm-start（`memory.py`） |
| **BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs** (ICML 2026) | rerank 的 `tournament` 图排序模式（`rerank.py`） |

```bibtex
@inproceedings{
agrawal2026blitzrank,
title={BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs},
author={Sheshansh Agrawal and Thien Hang Nguyen and Douwe Kiela},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=dc9si7lBTY}
}

@inproceedings{ICLR2026_b61f288d,
 author = {Hui, Yulong and Chen, Chao and Fu, Zhihang and Liu, Yihao and Ye, Jieping and Zhang, Huanchen},
 booktitle = {International Conference on Learning Representations},
 editor = {C. Vondrick and B. Hariharan and C. Raffel and L. Pinto and D. Yang and A. Faust},
 pages = {112154--112174},
 title = {Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval},
 url = {https://proceedings.iclr.cc/paper_files/paper/2026/file/b61f288da3c106f65d57b0d45b470b6b-Paper-Conference.pdf},
 volume = {2026},
 year = {2026}
}

@inproceedings{
chen2026jade,
title={{JADE}: Bridging the Strategic-Operational Gap in Dynamic Agentic {RAG}},
author={Yiqun Chen and Erhan Zhang and Tianyi Hu and Shijie Wang and Zixuan Yang and Meizhi Zhong and Xiaochi Wei and Yan Gao and Yi Wu and Yao Hu and Jiaxin Mao},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=KQT6pf4Mva}
}

@inproceedings{
wang2026gamrag,
title={{GAM}-{RAG}: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation},
author={Yifan Wang and Mingxuan Jiang and Zhihao Sun and Yixin Cao and Yicun Liu and Keyang Chen and Guangnan Ye and Hongfeng Chai},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=9YC7mbloXl}
}

@misc{sagar2026dynamicharnesssearchbuilding,
      title={Dynamic Harness Search: Building Multi-Agent Systems Per-Query via Prediction}, 
      author={Som Sagar and Shasha Li and Hejie Cui and Ransalu Senanayake and Sercan Ö. Arık},
      year={2026},
      eprint={2610.04137},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.04137}, 
}
```

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。
