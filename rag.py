"""检索 + 生成。

检索：向量和 BM25 各取前 TOPN，用 RRF（倒数排名融合）合并。
  - 问题里点名了公司 → 只在这些公司里检索，每家分到同样多的名额（跨公司对比题不会被一家占满）；
  - 问题里写了“半年/上半年”或“年报/全年/年度” → 只检索对应报告。
v2 改进（v2=False 即第一版，评测时两版都跑）：
  - 简称扩写：“归母净利润”等简称在财报表格里写的是全称，BM25 对不上，检索前补上全称；
  - 全景题：没点名公司、却问“哪家/哪几家/各家”的，改为 12 家每家各取 2 块，保证每家都有数据进上下文。
生成：把召回块编号 [1]..[k] 连同出处交给大模型，要求只依据资料作答并标注编号。
"""
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from index import COMPANIES, INDEX, QUERY_PREFIX, embedder, load_chunks, tokenize

ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')
TOPN = 50
RRF_K = 60
ALIASES = {'茅台': '贵州茅台', '汾酒': '山西汾酒', '洋河': '洋河股份', '古井': '古井贡酒', '老窖': '泸州老窖',
           '舍得': '舍得酒业', '酒鬼': '酒鬼酒', '迎驾': '迎驾贡酒', '口子': '口子窖'}
ALIASES.update({c: c for c in COMPANIES})
SYNONYMS = {'归母净利润': '归属于上市公司股东的净利润', '扣非净利润': '扣除非经常性损益的净利润',
            '经营现金流': '经营活动产生的现金流量净额', '营收': '营业收入'}
RE_PANORAMA = re.compile(r'哪家|哪几家|哪些公司|各家|\d+家')


@lru_cache(maxsize=1)
def _store():
    chunks = load_chunks()
    vecs = np.load(INDEX / 'vectors.npy')
    bm25 = pickle.load((INDEX / 'bm25.pkl').open('rb'))
    return chunks, vecs, bm25, embedder()


def detect_filters(q):
    companies = []
    for alias, name in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q and name not in companies:
            companies.append(name)
    report = None
    if re.search('半年|上半年|中报|1-6月|二季度', q):
        report = '2026半年报'
    elif re.search('年报|全年|年度|2025年', q):
        report = '2025年报'
    return companies, report


def _rank(scores, mask):
    s = np.where(mask, scores, -np.inf)
    top = np.argsort(-s)[:TOPN]
    return {int(i): r for r, i in enumerate(top) if np.isfinite(s[i])}


def expand(q):
    return q + ''.join(' ' + full for short, full in SYNONYMS.items() if short in q and full not in q)


def search(q, k=8, companies=None, report=None, mode='hybrid', v2=True):
    chunks, vecs, bm25, emb = _store()
    if companies is None and report is None:
        companies, report = detect_filters(q)
        if v2 and not companies and RE_PANORAMA.search(q):
            companies = list(COMPANIES)
    if v2:
        q = expand(q)
    qv = next(iter(emb.embed([QUERY_PREFIX + q])))
    vs = vecs @ (qv / np.linalg.norm(qv))
    bs = np.asarray(bm25.get_scores(tokenize(q)))
    groups = companies or [None]
    quota = max(2, -(-k // len(groups)))
    hits = []
    for comp in groups:
        mask = np.array([(comp is None or c['company'] == comp) and (report is None or c['report'] == report)
                         for c in chunks])
        rv = _rank(vs, mask) if mode != 'bm25' else {}
        rb = _rank(bs, mask) if mode != 'vec' else {}
        fused = {i: sum(1 / (RRF_K + r[i] + 1) for r in (rv, rb) if i in r) for i in set(rv) | set(rb)}
        for i in sorted(fused, key=lambda i: -fused[i])[:quota]:
            hits.append(dict(chunks[i], score=fused[i], vec_rank=rv.get(i), bm25_rank=rb.get(i)))
    return hits, dict(companies=companies, report=report)


def cite(h):
    return f"{h['company']}《{h['report']}》第{h['page']}页 · {h['section']} {h['sub']}".strip()


SYSTEM = """你是白酒行业上市公司财报问答助手。只能依据给出的财报片段回答，不要使用片段以外的知识。
要求：
- 数字照抄原文并写明单位（原文单位是“元”时可换算成亿元，但要保留原值）；涉及比较时先列出各家的数字再下结论。
- 每个事实后面用 [编号] 标注出自哪个片段，例如“营业收入 1,688.38 亿元[2]”。
- 片段里找不到的，明确写“所给资料中未找到”，不要猜。
- 用中文，简洁。"""


def answer(q, k=8, v2=True):
    from openai import OpenAI
    hits, filters = search(q, k, v2=v2)
    ctx = '\n\n'.join(f'[{n}] 出处：{cite(h)}\n{h["text"]}' for n, h in enumerate(hits, 1))
    client = OpenAI(api_key=os.environ['LLM_API_KEY'], base_url=os.environ['LLM_BASE_URL'], max_retries=1)
    r = client.chat.completions.create(
        model=os.environ.get('LLM_MODEL', 'deepseek-flash'), temperature=0, max_tokens=1500, timeout=90,
        messages=[{'role': 'system', 'content': SYSTEM},
                  {'role': 'user', 'content': f'财报片段：\n\n{ctx}\n\n问题：{q}'}],
        extra_body={'thinking': {'type': 'disabled'}})
    return dict(question=q, answer=r.choices[0].message.content.strip(), hits=hits, filters=filters)
