"""建索引：向量（bge-small-zh-v1.5，本地 onnx）+ BM25（jieba 分词）。

输入 data/chunks.jsonl，输出 data/index/{vectors.npy, bm25.pkl}。
每块送去编码的文字前面加上「公司 报告 章节 小节」，让“茅台的分红”这类问题能靠元数据命中。
"""
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

import jieba  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

ROOT = Path(__file__).parent
CHUNKS = ROOT / 'data' / 'chunks.jsonl'
INDEX = ROOT / 'data' / 'index'
EMB_MODEL = 'BAAI/bge-small-zh-v1.5'
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'  # bge 中文模型的检索指令
COMPANIES = ['贵州茅台', '五粮液', '泸州老窖', '山西汾酒', '洋河股份', '古井贡酒',
             '今世缘', '迎驾贡酒', '口子窖', '舍得酒业', '水井坊', '酒鬼酒']
for c in COMPANIES:
    jieba.add_word(c)
STOP = set('的 了 和 与 及 是 在 有 为 等 对 中 年 公司 多少 哪些 什么 如何 是否 情况 吗 呢 ？ ? ， , 。 、 | ： : （ ） ( ) - 《 》 “ ”'.split())


def tokenize(text):
    return [w for w in jieba.lcut(text.lower()) if w.strip() and w not in STOP]


def head(c):
    return f"{c['company']} {c['report']} {c['section']} {c['sub']}"


def load_chunks():
    return [json.loads(l) for l in CHUNKS.open(encoding='utf-8')]


def embedder():
    return TextEmbedding(EMB_MODEL, cache_dir=str(ROOT / 'models'))


def main():
    chunks = load_chunks()
    INDEX.mkdir(parents=True, exist_ok=True)
    docs = [head(c) + '\n' + re.sub(r'\s*\|\s*', ' ', c['text']) for c in chunks]
    print('BM25 ...')
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    pickle.dump(bm25, (INDEX / 'bm25.pkl').open('wb'))
    print('向量 ...')
    vecs = np.array(list(embedder().embed(docs, batch_size=64)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(INDEX / 'vectors.npy', vecs)
    print('完成', vecs.shape)


if __name__ == '__main__':
    main()
