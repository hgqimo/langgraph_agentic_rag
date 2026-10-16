# LangGraph Agentic RAG

[中文 README](README_zh.md)

An agentic RAG workflow built on **LangGraph**, combining an **INTERACT-RAG style
runtime retrieval loop**, a **GAM-RAG style training-free self-evolving memory**,
and retrieval-side extensions (multi-faceted retrieval, anchored matching,
model-level reranking). A single shared LLM backbone plays all agent roles; the
graph dynamically composes per-question workflows (`QR → R → DS → AG`, `QDS`,
`QDP`, ...), and retrieval experience is memorized across runs so the system gets
better *and* faster over turns, without any gradient updates.

## Highlights

- **9 agents, one backbone** — Planning, SubPlanning, QueryRewrite (QR),
  QueryDecomposition Parallel/Serial (QDP/QDS), DocumentSelection (DS),
  AnswerGeneration (AG), AnswerSummarization, EvidenceJudge — each with its own
  prompt, temperature and strict output parser.
- **Dynamic workflows** — PlanningAgent picks a chain shape per question;
  invalid plans fall back to a default workflow with a penalty (reward-compatible).
- **Runtime retrieval loop** — EvidenceJudge decides *at runtime* whether
  evidence is sufficient; if not, the query is rewritten with the missing-info
  context and retrieval repeats (bounded by `RAG_MAX_ROUNDS`).
- **Self-evolving memory (training-free)** — paragraph-level Kalman memory
  (attract useful passages, repel judged-irrelevant ones, persistent exclusion,
  memory-activated injection) + strategy-level `semantic_key → judge_plan`
  warm-start with a quadruple trust gate (cos / π / support / ≥3 independent
  successes) and capacity eviction. Feedback comes either from gold labels
  (`gold`, eval) or from the agents' own judge/DS decisions (`agent`, production).
- **Corpus Interaction Engine** — hybrid vector + BM25 + entity-boost retrieval
  with memory gating, used by all evaluations.
- **Model-level reranking (optional)** — the engine's 30-doc pool is reranked down
  to `top_k` by a cross-encoder, switchable between Qwen3-Reranker-4B (default)
  and NeoHorse-Jev-4B decision model (single-forward pool reranking, plus a
  tournament graph-sort mode).
- **Reproducible evaluation harness** — QASPER (single-hop over paper sections)
  and HotpotQA / 2WikiMultihopQA / MuSiQue (multi-hop, GAM-RAG *k-turn
  memorization* protocol, three arms: `off` / `agent` / `gold`).

## Workflow routing

PlanningAgent sees the question only and emits one tag inside
`<workflow>...</workflow>`; the chain is then fixed for the whole run. Three
routable shapes, seven effective tags:

| Workflow | Routed to | Example | Planner's prior |
|---|---|---|---|
| `QDS` | `decompose` (serial sub-questions) | "Which African country with the fastest economic growth since 1990 had the main export products in 2023?" | Sub-questions are dependent; each needs the previous answer |
| `QDP` | `decompose` (parallel sub-questions) | "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?" | Sub-questions are independent and can be searched at once |
| `AG` | `linear` | "3.5 squared?", or plain chitchat | Answerable from parametric knowledge; the corpus is not needed |
| `R,AG` | `linear` | "Which team is the 2025 NBA champion?" | Query is already retrieval-ready and the hit is expected to be unique |
| `R,DS,AG` | `linear` | "What was the fourth expansion pack for sims 2?" | Query is clear but the corpus holds many same-entity passages, so candidates need filtering |
| `QR,R,AG` | `linear` | "When did the lightning thief book come out?" | Query needs rewriting (informal title) yet the answer is unique once retrieved |
| `QR,R,DS,AG` | `linear` | "Who said \"i'm late i'm late for a very important date\"?" | Query is colloquial or ambiguous and candidates are expected to be noisy |

The legal set is every `{QR, R, DS, AG}` subset that keeps the QR→R→DS→AG
order and ends with AG (8 in total). `QR,AG`, `DS,AG` and `QR,DS,AG` pass the
order check but the prompt rules them out (DS requires R before it), so the
planner effectively picks among the seven above. An invalid tag falls back to
`RAG_DEFAULT_WORKFLOW` (default `QDS`) with a format penalty. Sub-questions get
a shorter menu: SubPlanningAgent only allows `R`, `DS`, `AG`.

Planner's priors, in the order it weighs them:

1. **Does the corpus matter at all** — chitchat, arithmetic and facts the model
   already holds go to `AG`; anything else keeps at least `R`.
2. **Is the query retrieval-ready** — colloquial quotes, informal titles and
   context-dependent phrasing need `QR` to normalize first.
3. **How noisy are the candidates expected to be** — hot entities ("sims 2"),
   broad questions ("summarize this paper") and multi-source topics imply many
   same-entity passages, so `DS` earns its call; narrow facts ("2025 champion")
   do not.
4. **Do the sub-questions depend on each other** — dependent → `QDS`,
   independent → `QDP`, otherwise stay linear.

Two things soften a wrong static pick: AG receives all candidates and filters
them itself (every document goes into the prompt, see `agents.py`), and the
runtime judge loop re-retrieves with the missing-info context when the evidence
is judged insufficient.

## Headline results

| Setting (40–60 questions, 5 turns, temp=0, Qwen 27B judge) | Outcome |
|---|---|
| QASPER single-hop, `agent` feedback | Judge-Acc **+11pp** vs 0-turn |
| HotpotQA, `gold` feedback | Judge-Acc **0.925 (+3pp)**, F1 **+5pp**, retrieval rounds **−16%** — both curves monotonic |
| 2WikiMultihopQA | neutral (high baseline, small corpus); answer-format fix alone lifted baseline **+15pp** |
| MuSiQue 2–4 hop, `agent` feedback | **−7.5pp** negative evolution — self-annotated feedback is unreliable on long chains (warm-start pollution, root-caused via ablation) |

**Takeaway**: self-evolution pays off as `f(feedback quality × task difficulty)`.
Gold feedback is safe everywhere; agent self-annotation works up to 2 hops and
needs the strategy channel disabled (or GT-level feedback) beyond that.

## Gains from each component

Measured on 40-question / 5-turn runs at temp=0 (`agent` or `gold` feedback).

| Component | Source | Reported gain |
|---|---|---|
| Multi-faceted retrieval + anchored matching | Interact-RAG | single-round evidence hit@retrieval 0.47 → 0.53–0.63 |
| Runtime retrieval loop (EvidenceJudge) | Interact-RAG | QASPER F1 0.25 → 0.48 (temp=0, 14 questions) and evidence hit 0.21 → 0.57, at 2.1 retrieval rounds instead of 1.0 — the largest single gain in this repo |
| Self-evolving memory (doc + strategy levels) | GAM-RAG | `gold`, 5 turns: evidence hit **+16.5pp** (0.575 → 0.740), retrieval rounds **−14%**, F1 +3.6pp. `agent` (no gold labels), 10 turns: evidence hit **+11.0pp**, rounds −12%, hit climbing monotonically to 0.750 |
| Strategy warm-start | GAM-RAG | 40/40 questions warm-started by turn 10 (`agent`); overtakes doc-level-only memory from turn 5 (0.775 vs 0.750) |
| Model-level reranking | BlitzRank + cross-encoder | MuSiQue 300 unseen questions, same top-30 candidate pool: hit@8 **0.9867** (Qwen3-Reranker-4B) vs 0.9600 (Jev-4B), mrr@8 0.846 vs 0.714; nDCG@10 **0.82** on a 1000-doc pool. Optional, default `off` |
| Cost-aware execution reward | Dynamic Harness Search | reward = task score − resource penalties, rescaled into a utility; 14 real trajectories match the upstream training loop on **0** mismatched fields, so the judge agent drops straight into PPO |

Two honest limits: the single-round retrieval gains do not reach end-to-end F1
(QASPER stays in the 0.29–0.30 band — the bottleneck moved to answer
generation), and on MuSiQue's 2–4 hop chains `agent` feedback loses 7.5pp
because self-annotated signals are unreliable there (see Headline results).

## Repository layout

```
.
├── src/langgraph_rag/        # core package
│   ├── config.py             # env-driven configuration (see .env.example)
│   ├── state.py              # RagState / ChainState / TraceStep
│   ├── agents.py             # 9 agents: prompts + strict parsers
│   ├── graph.py              # LangGraph topology (plan → chain → retrieval loop)
│   ├── runtime.py            # agent invocation, memory feedback attribution
│   ├── engine.py             # corpus interaction engine (vector+BM25+entity+memory)
│   ├── memory.py             # self-evolving memory (Kalman, doc + strategy levels)
│   ├── vector.py             # embedding cache + cosine retriever
│   ├── rerank.py             # model-level reranking (Qwen3 cross-encoder / Jev)
│   ├── jev_decisions.py      # Jev decisions client (OpenAI Decisions API shaped)
│   ├── llm.py                # OpenAI-compatible chat model + scripted mock
│   ├── reward.py             # F1 / EM / judge metrics
│   └── tools.py              # retrieval tool interface
├── eval/
│   ├── qasper_eval.py        # single-hop evaluation (QASPER)
│   └── multihop_eval.py      # multi-hop k-turn memorization (GAM-RAG protocol)
├── scripts/
│   ├── run.py                # CLI entry: single question / batch / mermaid graph
│   └── walkthrough.py        # step-by-step trace of one query
├── datasets/                 # bring-your-own data (see datasets/README.md)
├── results/                  # evaluation outputs (gitignored)
├── .env.example              # all configuration knobs with defaults
└── requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
cp .env.example .env                 # edit endpoints/models, then:
set -a; source .env; set +a
```

Requirements: Python ≥ 3.10, an OpenAI-compatible LLM endpoint (e.g. vLLM) and
optionally an embeddings endpoint. Everything also runs fully offline in mock
mode (no services needed).

## Quickstart

```bash
# Offline mock run — validates graph topology, no LLM/retrieval service needed
python scripts/run.py --mock --mock-route qds \
    --question "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?" \
    --gold "mock final answer"

# Real backend (vLLM) with local corpus + adaptive retrieval loop + memory
python scripts/run.py --base-url http://127.0.0.1:8000/v1 --model <model> \
    --retrieval-mode engine --corpus corpus.jsonl --max-rounds 3 \
    --question "..." --gold "..." --show-process

# Print the graph as Mermaid
python scripts/run.py --mermaid

# Offline self-test of the memory machinery (Kalman dynamics, gating,
# exclusion, injection, strategy warm-start, defer/commit, persistence)
PYTHONPATH=src python -m langgraph_rag.memory
```

## Evaluation

Download datasets first (layout and sources in [`datasets/README.md`](datasets/README.md)).

```bash
# Single-hop (QASPER): build corpus from 12 papers, evaluate 24 questions
python eval/qasper_eval.py --base-url http://127.0.0.1:8000/v1 --model <model> \
    --embed-url http://127.0.0.1:10200/v1 --papers 12 --questions 24 \
    --workers 4 --scope open --out results/qasper.json

# Multi-hop, GAM-RAG k-turn memorization, three arms
python eval/multihop_eval.py --dataset hotpotqa --questions 40 --turns 5 \
    --arms off,agent,gold --base-url http://127.0.0.1:8000/v1 --model <model> \
    --workers 3 --temperature 0 --out results/mh_hotpotqa.json
# --dataset: hotpotqa | 2wiki | musique ; --arms: off | agent | gold
```

Memory mode is controlled by `RAG_MEMORY_FEEDBACK` (`off`/`gold`/`agent`) and
`RAG_MEMORY_DB` (SQLite persistence for cross-turn evolution); the eval
scripts wire these per arm automatically.

## References

The implementation aligns with the following work. Each paper's landing spot in
the code is listed.

| Paper | Where it lands in this repo |
|---|---|
| **JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG** (ICML 2026) | Graph topology and per-question workflow composition (`graph.py`), the 9 agent prompts and strict parsers (`agents.py`), metric definitions and per-agent step aggregation (`reward.py`) |
| **Dynamic Harness Search: Building Multi-Agent Systems Per-Query via Prediction** (arXiv 2026) | The reward follows its cost-aware execution reward. Here that is `penalty` + `final_reward` → `whole`/`step_reward` in `reward.py`, with EM/F1 as the task score and cost charged on format validity plus turn/retrieval counts. Its per-query harness predictor is not implemented |
| **Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval** (ICLR 2026) | Multi-faceted retrieval, anchored matching and context shaping in the Corpus Interaction Engine (`engine.py`); the EvidenceJudge runtime retrieval loop (`graph.py`) |
| **GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval** (ICML 2026) | Paragraph-level Kalman memory, persistent exclusion, memory-activated injection and strategy warm-start (`memory.py`) |
| **BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs** (ICML 2026) | The `tournament` graph-sort mode of reranking (`rerank.py`) |

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

## License

Apache-2.0. See [LICENSE](LICENSE).
