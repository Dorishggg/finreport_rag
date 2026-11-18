"""PDF → 带元数据的切块（data/chunks.jsonl）。

每页先用 pdfplumber 找表格，表格按行列还原成 markdown；表格区域以外的文字按行抽出。
文字和表格按纵坐标排序后顺序处理，同时跟踪“第X节”（章节）和“一、……”（小节）标题。
切块规则：
- 文字：同一小节内按行累加，满 CHUNK_CHARS 字切一块；
- 表格：每块重复表头，按行累加到 TABLE_CHARS 字；跨页续表沿用上一页的表头。
每块都带 公司 / 报告 / 章节 / 小节 / 页码。
"""
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).parent
PDF_DIR = ROOT / 'data' / 'pdf'
OUT = ROOT / 'data' / 'chunks.jsonl'
CHUNK_CHARS = 500
TABLE_CHARS = 900

CN = '一二三四五六七八九十'
RE_SECTION = re.compile(rf'^第[{CN}]+节\s*(.+)$')
RE_SUB = re.compile(rf'^[{CN}]+、\s*(.{{2,40}})$')
RE_PAGENO = re.compile(r'^[\d\s/-]+$')
RE_TOC = re.compile(r'\.{6,}|…{3,}')
RE_NOCAP = re.compile(r'^(单位|币种|□|√)')


def clean_cell(c):
    return re.sub(r'\s*\n\s*', '', c or '').strip()


def table_rows(tb):
    rows = [[clean_cell(c) for c in r] for r in tb.extract()]
    rows = [r for r in rows if any(r)]
    if not rows:
        return rows
    # 去掉整列为空的列
    keep = [j for j in range(len(rows[0])) if any(j < len(r) and r[j] for r in rows)]
    rows = [[r[j] if j < len(r) else '' for j in keep] for r in rows]
    # 深交所报告常见的错位网格：没有任何一行填满所有列 → 每行只保留非空格子
    if max(sum(map(bool, r)) for r in rows) < len(keep):
        rows = [[c for c in r if c] for r in rows]
    # 单元格内换行被拆成多行：只有一个文字格子的行并回上一行的行名（“分产品”这类分组行不并）
    out = []
    for r in rows:
        cells = [c for c in r if c]
        if (out and len(cells) == 1 and len(keep) > 2 and not re.search(r'\d', cells[0])
                and not cells[0].endswith('：') and len(out[-1][0]) >= 8):  # 被折行的行名通常已写满列宽
            out[-1][0] += cells[0]
        else:
            out.append(r)
    width = max(map(len, out))
    # 压缩后表头常缺左上角那一格 → 表头左侧补空；数据行缺的是末尾的空格子 → 右侧补空
    return [[''] * (width - len(out[0])) + out[0]] + [r + [''] * (width - len(r)) for r in out[1:]]


def md_row(r):
    return '| ' + ' | '.join(r) + ' |'


def page_elements(page):
    """返回本页按纵坐标排好序的元素：('line', top, text) / ('table', top, rows)。"""
    tables = page.find_tables()
    bboxes = [t.bbox for t in tables]

    def outside(obj):
        cx, cy = (obj['x0'] + obj['x1']) / 2, (obj['top'] + obj['bottom']) / 2
        return not any(x0 <= cx <= x1 and t <= cy <= b for x0, t, x1, b in bboxes)

    els = []
    for ln in page.filter(lambda o: o.get('object_type') != 'char' or outside(o)).extract_text_lines():
        els.append(('line', ln['top'], ln['text'].strip()))
    for t in tables:
        rows = table_rows(t)
        if rows:
            els.append(('table', t.bbox[1], rows))
    return sorted(els, key=lambda e: e[1])


def parse_pdf(path):
    code, company, report = path.stem.split('_')
    meta = dict(code=code, company=company, report=report)
    chunks, buf = [], []
    state = dict(section='', sub='', buf_page=None)
    last_table = None  # 上一个表格的 (列数, 表头)，用于跨页续表

    def emit(kind, text, p0, p1):
        chunks.append(dict(meta, id=f'{code}_{report}_{len(chunks):05d}', type=kind,
                           section=state['section'], sub=state['sub'], page=p0, page_end=p1, text=text))

    def flush(page_no):
        if buf:
            emit('text', '\n'.join(buf), state['buf_page'], page_no)
            buf.clear()

    with pdfplumber.open(path) as pdf:
        n = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            p = i + 1
            els = page_elements(page)
            lines = [e[2] for e in els if e[0] == 'line']
            if sum(bool(RE_TOC.search(l)) for l in lines) >= 3:  # 目录页
                continue
            caption = ''
            for j, e in enumerate(els):
                if e[0] == 'line':
                    text = e[2]
                    if not text or RE_PAGENO.match(text) or (j == 0 and '报告' in text and len(text) < 40):
                        continue  # 页码、页眉
                    m = RE_SECTION.match(text)
                    if m:
                        flush(p)
                        state['section'], state['sub'] = text.replace(' ', ''), ''
                        last_table = None
                        continue
                    m = RE_SUB.match(text)
                    if m:
                        flush(p)
                        state['sub'] = text.replace(' ', '')
                    if not buf:
                        state['buf_page'] = p
                    buf.append(text)
                    if not RE_NOCAP.match(text):
                        caption = text
                    if sum(map(len, buf)) >= CHUNK_CHARS:
                        flush(p)
                else:
                    rows = e[2]
                    ncol = len(rows[0])
                    cont = j == 0 and last_table and last_table[0] == ncol and rows[0] != last_table[1]
                    head = last_table[1] if cont else rows[0]
                    body = rows if cont else rows[1:]
                    flush(p)
                    title = (last_table[2] + '（续表）') if cont else caption
                    last_table = (ncol, head, title)
                    group, size = [], 0
                    for r in body or [[]]:
                        group.append(r)
                        size += sum(map(len, r))
                        if size >= TABLE_CHARS:
                            emit('table', '\n'.join([f'表：{title}', md_row(head)] + [md_row(x) for x in group if x]), p, p)
                            group, size = [], 0
                    if any(group):
                        emit('table', '\n'.join([f'表：{title}', md_row(head)] + [md_row(x) for x in group if x]), p, p)
                    caption = ''
            if els and els[-1][0] != 'table':
                last_table = None
        flush(n)
    print(f'{path.name}: {n} 页 → {len(chunks)} 块', flush=True)
    return chunks


def main():
    files = sorted(PDF_DIR.glob('*.pdf'))
    if len(sys.argv) > 1:
        files = [f for f in files if any(k in f.name for k in sys.argv[1:])]
    with Pool(6) as pool:
        results = pool.map(parse_pdf, files)
    with OUT.open('w', encoding='utf-8') as f:
        for chunks in results:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print('共', sum(map(len, results)), '块 →', OUT)


if __name__ == '__main__':
    main()
