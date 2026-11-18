# 白酒上市公司财报问答知识库（第三周作业 · 方向 A）

12 家白酒上市公司的 2025 年报和 2026 半年报（共 24 份、3,830 页），切成 14,635 块，建成“向量 + BM25”混合检索的问答页面，答案标注出处（公司 · 报告 · 页码 · 章节）。

公司：贵州茅台、五粮液、泸州老窖、山西汾酒、洋河股份、古井贡酒、今世缘、迎驾贡酒、口子窖、舍得酒业、水井坊、酒鬼酒。

## 流程

| 步骤 | 文件 | 做法 |
|---|---|---|
| 下载 | `download.py` | 调用巨潮资讯公告查询接口，下载年报 / 半年报全文 PDF（跳过摘要和英文版）到 `data/pdf/`。PDF 不放进仓库 |
| 解析切块 | `parse.py` | 用 pdfplumber 找表格，表格按行列还原成 markdown（删除空列、修正深交所的错位网格、合并单元格内换行拆出来的续行、跨页续表沿用上一页表头），表格以外的文字按行抽取；同时识别“第X节”和“一、……”标题。正文满 500 字切一块；表格每块都重复表头，按 900 字切。每块都带公司 / 报告 / 章节 / 小节 / 页码 |
| 建索引 | `index.py` | 向量：`BAAI/bge-small-zh-v1.5`（fastembed 本地 onnx，CPU 可跑）；BM25：jieba 分词，公司名作为自定义词。送去编码的文字前面加上“公司 报告 章节 小节” |
| 检索与作答 | `rag.py` | 两路各取前 50，用 RRF 融合；从问题里识别公司和报告，先过滤；点名多家公司时按公司平均分配名额。v2 加了简称扩写（如“归母净利润”→ 全称）和全景题按公司分配名额。把召回的 8 块编号后交给大模型，要求每句话标注 [编号] |
| 页面 | `app.py` | FastAPI 页面：显示答案、出处列表（点击直接打开 PDF 对应页）、原文片段、每块在向量 / BM25 两路里的排名 |
| 评测 | `evaluate.py` + `questions.json` | 10 道题（3 道跨公司，其中 2 道全景题），第一版和 v2 各跑一遍，记录召回块、标准出处页在向量 / BM25 / 混合三种检索里的排名 |

## 运行

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe pdfplumber pymupdf jieba rank_bm25 fastembed fastapi uvicorn openai python-dotenv requests numpy
cp .env.example .env          # 填入任意 OpenAI 兼容接口的 key / base_url / model
python download.py            # 约 40 MB
python parse.py               # 约 4 分钟（6 进程）
python index.py               # 向量模型从 hf-mirror 下载；编码约 5 分钟
python -m uvicorn app:app --port 8001   # 打开 http://127.0.0.1:8001
python evaluate.py
```

key 只放在 `.env` 里，`.env` 已经写进 `.gitignore`。

## 交付物

- `screenshots/`：问答页面截图（Q1 单公司题、Q8 点名对比题、Q10 全景题）
- `results/逐题记录.md`：10 道题逐题记录：召回了哪些块、答对没有、错在哪
- `results/eval_retrieval.md`：标准出处页在向量 / BM25 / 混合检索中的排名
- `results/一页结论.pdf`：一页结论

## 结论摘要

第一版答对 6/10，错题全部集中在“原因解释”和“跨公司”两类。单公司的查找题检索准、出处清楚；全景题（“12 家里哪家最……”）用 top-k 检索不能保证每家都拿到数据，模型还会根据不全的资料下确定结论。v2 给每家分配名额后只改善了一半。全景题应该先把关键指标抽成结构化表再回答。详见 `results/一页结论.pdf`。
