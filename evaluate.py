"""逐题评测：python evaluate.py

对 questions.json 里的每道题：
- 混合检索 + 大模型作答，记录答案和召回的 8 块（公司/报告/页码/类型/向量排名/BM25 排名）；
- 另外只用向量、只用 BM25 各检索一次，看标准出处页有没有被召回。
第一版（v2=False）和改进版（v2=True）各跑一遍。
输出 results/eval_raw_v1.json、eval_raw_v2.json 和 eval_retrieval.md；答对与否由人工对照 gold 判定，写在 results/逐题记录.md。
"""
import json
from pathlib import Path

from rag import ROOT, answer, cite, search

OUT = ROOT / 'results'


def gold_rank(hits, gold_src):
    """标准出处在召回列表里的最好名次（1 起），没召回为 None。"""
    gold = {tuple(g) for g in gold_src}
    for n, h in enumerate(hits, 1):
        if (h['company'], h['report'], h['page']) in gold:
            return n
    return None


def run(qs, v2):
    raw, md = [], ['| # | 题型 | 向量 | BM25 | 混合 | 召回块数 | 检索范围 |', '|---|---|---|---|---|---|---|']
    for q in qs:
        res = answer(q['q'], v2=v2)
        ranks = {m: gold_rank(search(q['q'], mode=m, v2=v2)[0], q['gold_src']) for m in ('vec', 'bm25')}
        ranks['hybrid'] = gold_rank(res['hits'], q['gold_src'])
        hits = [dict(n=n, src=cite(h), type=h['type'], vec_rank=h['vec_rank'], bm25_rank=h['bm25_rank'],
                     text=h['text'][:160]) for n, h in enumerate(res['hits'], 1)]
        raw.append(dict(q, answer=res['answer'], filters=res['filters'], gold_rank=ranks, hits=hits))
        f = res['filters']
        fmt = lambda r: f'第{r}' if r else '未召回'
        md.append(f"| {q['id']} | {q['type']} | {fmt(ranks['vec'])} | {fmt(ranks['bm25'])} | {fmt(ranks['hybrid'])} | {len(hits)} | "
                  f"{'、'.join(f['companies']) or '全部'} / {f['report'] or '全部报告'} |")
        print('v2' if v2 else 'v1', q['id'], ranks, flush=True)
    (OUT / f"eval_raw_{'v2' if v2 else 'v1'}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding='utf-8')
    return md


def main():
    OUT.mkdir(exist_ok=True)
    qs = json.loads((ROOT / 'questions.json').read_text(encoding='utf-8'))
    parts = ['# 检索命中：标准出处页在召回列表中的名次', '']
    for v2, title in ((False, '第一版（混合检索，每题 8 块）'), (True, '改进版 v2（简称扩写 + 全景题每家 2 块）')):
        parts += [f'## {title}', ''] + run(qs, v2) + ['']
    (OUT / 'eval_retrieval.md').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()
