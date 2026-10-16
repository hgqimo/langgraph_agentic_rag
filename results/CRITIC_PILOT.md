# RAG-Critic 式答案批评集成 + 试点报告（2026-10-15）

## 集成内容（已完成，默认关闭）

把 RAG-Critic（ACL 2025, RUC）的 critic-guided 方法论集成进 langgraph_rag，
遵守两条约束：**不训练模型**、**错误类型用 Jev 判定**。

| 组件 | 文件 | 内容 |
|---|---|---|
| 标签体系 + 批评器 | `src/langgraph_rag/critic.py`（新） | 7 tag1 / 19 tag2 层级标签（内嵌自 `all_tags_structure_final.json`）；`JevCritic`：Judgement（2 候选 Choice）+ tag2 分类（19 候选 Choice），各 1 次 prefill-only forward，~145ms；失败安全（服务不在→None 透传） |
| 配置 | `config.py` | `critic_mode`（默认 **off**）/`critic_max_rounds=1`/`critic_min_prob=0.5`/`critic_max_doc_chars=240`/`critic_max_docs=5` |
| 状态 | `state.py` | ChainState 增 `critic_round`/`critic_instruction`/`critic_verdict` |
| 图节点 | `graph.py` | `answer → critic →` 路由：检索类错误（Incomplete/Irrelevant/Erroneous Information）→ 错误类型注入 judge_missing/why → `loop_rewrite` 重写 query 再检索（复用 INTERACT-RAG 回路）；生成类错误 → 带纠错指令重答 `answer`；预算耗尽降级为接受（修过一个路由/状态不一致 bug） |
| answer 节点 | `agents.py` | `build_answer_messages(additional_instruction=)`，纠错指令注入 user 末端（贴近生成位置） |
| Runtime | `runtime.py` | `__post_init__` 挂 JevCritic（服务不健康则禁用并告警） |
| eval | `eval/multihop_eval.py` | `--critic-mode off\|on`；`qasper_eval.evaluate_one` 行增 `critic` 裁决字段 |

未采用：RAG-Critic 的 plan_agent 自由 Python 代码生成+exec（脆且重）、tag3
开放集（2000+，超出 Jev 255 候选契约）、RAG-Critic-3B 训练管线。

冒烟全过：图逻辑（检索类→loop_rewrite→再检索→重答；生成类→带指令重答；
预算耗尽降级）、真 Jev 单元（判对 p=0.0003 / 判错 p=0.9858 且分类合理）、
失败安全（服务不可用无批评透传）。

## 试点（musique 30 题，turns=1，agent 臂，同 seed 双跑）

数据：`results/pilot_critic_off.json` / `pilot_critic_on.json`。

| 指标 | OFF | ON | Δ |
|---|---|---|---|
| judge_acc | 0.6667 | 0.6333 | -0.033 |
| contain_acc | 0.6667 | 0.6333 | -0.033 |
| answer_f1 | 0.6878 | 0.6611 | -0.027 |
| hitR | 1.000 | 1.000 | 0 |
| mean_retrieval_times | 4.23 | 4.37 | +0.13 |

**Judgement 信号质量（按题聚合）**：判错题 21/30（70%）；答案错被 flag
（召回）9/12 = 75%；答案对未 flag（纯净）仅 6/18 = 33%——**误报 12/18**。
裁决分布：101 次子链裁决，58% 判错；路由 32 regenerate / 27 retrieve；
tag1 以 Incomplete Information(16)/Off-Topic(14)/Inaccurate(13) 为主。

## 结论：机制成立，信号不足，默认保持 off

1. **工程集成正确**（冒烟三级全过，路由/预算/失败安全符合设计）；
2. **Judgement 过度触发**：误报主因是**粒度错配**——critic 评判的是子链
   中间答案（每题 ~3.4 次），而 EM 度量最终答案；中间答案"不完整"是正常
   形态。次要因素：Jev state 2048-token 预算内只看得见前 5 篇 ×240 字符
   截断文档，对证据视图不完整易判"检索不足"；
3. **纠错收益为负**（噪声级 -3pt）：regenerate 同文档重答倾向给出相同答案
   （出现同 p 重复 flag）；retrieve 路线同 query 再检索增量有限；
4. 与 rerank 的教训一致：**多轮 agent 会吸收/稀释单点增益**，答案级纠错
   的增益窗口比预期窄。

## 下一步建议（按优先级）

1. **最终答案级 critic**（改动小）：把批评节点从子链移到主图 `summarize`
   之后，每题只评判一次最终 prediction——直接对齐 RAG-Critic 的粒度，
   预期误报率大幅下降；若信号达标（FP<20%）再谈自主纠错；
2. **信号化使用**：Judgement 不作自主纠错，而作拒答触发/记忆负反馈特征
   （" critic 判错且 p>0.9 → 标记低质轨迹"）；
3. 阈值/预算调优（min_prob 0.5→0.8）优先级最低——误报多为高置信（p>0.9），
   调阈值治不了标定问题。
