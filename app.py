"""问答页面：python -m uvicorn app:app --port 8001，浏览器打开 http://127.0.0.1:8001"""
import html
import re
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from rag import ROOT, answer, cite

app = FastAPI()
PDF_DIR = ROOT / 'data' / 'pdf'

CSS = """
body{font-family:"Microsoft YaHei",sans-serif;max-width:1000px;margin:24px auto;color:#222;background:#fafafa}
h1{font-size:22px;margin-bottom:4px} .sub{color:#777;font-size:13px;margin-bottom:16px}
form{display:flex;gap:8px} input{flex:1;font-size:16px;padding:8px 10px;border:1px solid #bbb;border-radius:6px}
button{font-size:16px;padding:8px 18px;border:0;border-radius:6px;background:#8b1a1a;color:#fff;cursor:pointer}
.ans{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:18px 0;white-space:pre-wrap;line-height:1.7}
.ans sup a{color:#8b1a1a;text-decoration:none;font-weight:bold}
.flt{font-size:13px;color:#555;margin-top:14px}
.hit{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:8px 12px;margin:8px 0;font-size:14px}
.hit .src{font-weight:bold} .hit .src a{color:#1a4d8b} .tag{font-size:12px;color:#888;margin-left:6px}
pre{white-space:pre-wrap;font-size:12.5px;background:#f6f6f6;padding:8px;border-radius:4px;max-height:260px;overflow:auto}
"""


def render(res):
    hits = res['hits']
    ans = html.escape(res['answer'])
    ans = re.sub(r'\[(\d+)\]', r'<sup><a href="#s\1">[\1]</a></sup>', ans)
    f = res['filters']
    flt = f"检索范围：公司 = {'、'.join(f['companies']) or '全部 12 家'}；报告 = {f['report'] or '年报 + 半年报'}"
    items = []
    for n, h in enumerate(hits, 1):
        pdf = quote(f"{h['code']}_{h['company']}_{h['report']}.pdf")
        rank = f"向量第{h['vec_rank'] + 1 if h['vec_rank'] is not None else '-'} · BM25第{h['bm25_rank'] + 1 if h['bm25_rank'] is not None else '-'}"
        items.append(
            f'<div class="hit" id="s{n}"><span class="src">[{n}] <a href="/pdf/{pdf}#page={h["page"]}" target="_blank">'
            f'{html.escape(cite(h))}</a></span><span class="tag">{"表格" if h["type"] == "table" else "正文"} · {rank}</span>'
            f'<details><summary>原文片段</summary><pre>{html.escape(h["text"])}</pre></details></div>')
    return f'<div class="ans">{ans}</div><div class="flt">{flt}</div>' + ''.join(items)


@app.get('/', response_class=HTMLResponse)
def index(q: str = ''):
    body = render(answer(q)) if q.strip() else ''
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>白酒财报问答</title><style>{CSS}</style></head>
<body><h1>白酒上市公司财报问答</h1>
<div class="sub">12 家公司 · 2025 年报 + 2026 半年报 · 向量（bge-small-zh）+ BM25 混合检索 · 答案标注出处页码</div>
<form><input name="q" value="{html.escape(q)}" placeholder="例如：贵州茅台2025年的营业收入和净利润是多少？"><button>提问</button></form>
{body}</body></html>"""


@app.get('/pdf/{name}')
def pdf(name: str):
    path = (PDF_DIR / name).resolve()
    if path.parent != PDF_DIR.resolve() or not path.exists():
        return HTMLResponse('not found', status_code=404)
    return FileResponse(path, media_type='application/pdf')
