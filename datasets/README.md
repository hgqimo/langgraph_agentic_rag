# Datasets

```
datasets/
├── qasper/
│   └── raw/qasper-dev-v0.3.json          # QASPER dev split
├── HotpotQA/
│   └── raw/hotpot_dev_distractor_v1.json # HotpotQA dev (distractor)
├── 2WikiMultihopQA/
│   └── dev.json                          # 2WikiMultihopQA dev
└── MuSiQue/
    └── musique_ans_v1.0_dev.jsonl        # MuSiQue-ans dev
```

## 下载来源

| 数据集 | 来源 | 文件 |
|---|---|---|
| QASPER | https://github.com/allenai/qasper-dev-splits-annotated （或 https://ai2-systems.github.io/qasper/ ） | `qasper-dev-v0.3.json` |
| HotpotQA | https://hotpotqa.github.io/ | `hotpot_dev_distractor_v1.json` |
| 2WikiMultihopQA | https://github.com/Alab-NII/2wikimultihop | `dev.json` |
| MuSiQue | https://github.com/StonyBrookNLP/musique （需签署数据协议后获取 `musique_ans_v1.0_dev.jsonl`） | `musique_ans_v1.0_dev.jsonl` |

示例（QASPER / HotpotQA）：

```bash
mkdir -p datasets/qasper/raw datasets/HotpotQA/raw
wget -O datasets/qasper/raw/qasper-dev-v0.3.json \
  https://raw.githubusercontent.com/allenai/qasper-dev-splits-annotated/main/data/qasper-dev-v0.3.json
wget -O datasets/HotpotQA/raw/hotpot_dev_distractor_v1.json \
  https://raw.githubusercontent.com/hotpotqa/hotpot/master/data/hotpot_dev_distractor_v1.json
```

## 语料构建协议

评测脚本按 HippoRAG2 协议从采样题目自带的 context 合并去重构建共享检索语料
（无需额外下载全量语料库）：每题的 `context`（HotpotQA/2Wiki）或
`paragraphs`（MuSiQue）段落进入向量索引，`supporting_facts` /
`is_supporting` 标注作为证据命中（hitR/hitAG）金标。
