"""从巨潮资讯下载白酒行业 12 家公司的 2025 年报 + 2026 半年报全文 PDF 到 data/pdf/。

用法：python download.py
"""
import os
import re
import time
from pathlib import Path

import requests

for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

H = {'User-Agent': 'Mozilla/5.0'}
OUT = Path(__file__).parent / 'data' / 'pdf'
COMPANIES = ['600519', '000858', '000568', '600809', '002304', '000596',
             '603369', '603198', '603589', '600702', '600779', '000799']
# (报告标签, 标题里的年份关键词, 公告分类)
REPORTS = [('2025年报', '2025年年度报告', 'category_ndbg_szsh'),
           ('2026半年报', '2026年半年度报告', 'category_bndbg_szsh')]
SKIP = re.compile('摘要|英文|取消|已取消')


def stock_index():
    d = requests.get('http://www.cninfo.com.cn/new/data/szse_stock.json', headers=H, timeout=30).json()
    return {s['code']: s for s in d['stockList']}


def query(code, org_id, category):
    column = 'sse' if code.startswith('6') else 'szse'
    r = requests.post('http://www.cninfo.com.cn/new/hisAnnouncement/query', headers=H, timeout=30, data=dict(
        stock=f'{code},{org_id}', tabName='fulltext', pageSize=50, pageNum=1, column=column,
        category=category, seDate='2026-01-01~2026-12-31', isHLtitle='true'))
    return r.json().get('announcements') or []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = stock_index()
    for code in COMPANIES:
        s = idx[code]
        name = s['zwjc'].replace(' ', '')
        for label, key, cat in REPORTS:
            dst = OUT / f'{code}_{name}_{label}.pdf'
            if dst.exists():
                continue
            anns = [a for a in query(code, s['orgId'], cat)
                    if key in re.sub('<.*?>', '', a['announcementTitle']) and not SKIP.search(a['announcementTitle'])]
            if not anns:
                print('未找到', code, name, label)
                continue
            a = max(anns, key=lambda a: a['announcementTime'])  # 有修订版时取最新一份
            pdf = requests.get('http://static.cninfo.com.cn/' + a['adjunctUrl'], headers=H, timeout=120).content
            dst.write_bytes(pdf)
            print(f'{dst.name}  {len(pdf) / 1e6:.1f} MB')
            time.sleep(1)


if __name__ == '__main__':
    main()
