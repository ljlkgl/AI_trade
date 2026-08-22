# -*- coding: utf-8 -*-
"""从币安公开接口分页拉取 SOLUSDT 30m K线，存为与现有数据同格式 CSV。"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'SOLUSDT_30m_kline.csv')

BASE = 'https://api.binance.com/api/v3/klines'
INTERVAL = '30m'
LIMIT = 1000

def fetch(start_ms):
    q = urllib.parse.urlencode({'symbol': 'SOLUSDT', 'interval': INTERVAL,
                                'startTime': start_ms, 'limit': LIMIT})
    url = f"{BASE}?{q}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def main():
    rows = []
    start = 1508198400000  # 2017-10-17 (SOL 上线后尽早)
    seen = set()
    t0 = time.time()
    cur = start
    while True:
        data = fetch(cur)
        if not data:
            break
        new = 0
        for k in data:
            ts = k[0]
            if ts in seen:
                continue
            seen.add(ts)
            rows.append((ts,
                         k[1], k[2], k[3], k[4], k[5],
                         k[8], k[9], k[10]))  # o,h,l,c,vol,trades,tb,ts,...
            new += 1
        last = data[-1][0]
        cur = last + 1
        if new == 0:
            # 如果没新增且已到最近，停止
            if time.time() - t0 > 600:
                break
        # 停止条件：达到当前时间附近
        if last >= int(time.time() * 1000) - 1800000:
            break
        time.sleep(0.15)  # 限速
        if len(rows) > 2_000_000:
            break
    print(f"拉取 {len(rows)} 根 30m K线, 耗时 {time.time()-t0:.1f}s")

    # 排序并写 CSV（按现有格式：datetime,o,h,l,c,volume,trades,taker_buy,taker_sell）
    rows.sort(key=lambda x: x[0])
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('datetime,open,high,low,close,volume,trades,taker_buy,taker_sell\n')
        for r in rows:
            ts, o, h, l, c, v, tb, ts_, ts__ = r
            # k[10] 为 taker_buy_base_volume，这里用 k[8](taker buy quote 由 source 决定)
            # 简化：taker_buy 用币安 k[9] (taker buy base vol), trades=k[8]
            dt = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts / 1000))
            f.write(f"{dt},{o},{h},{l},{c},{v},{tb},{'0'},{'0'}\n")
    print(f"[已保存] {OUT}")
    print(f"跨度: {rows[0][0]} ~ {rows[-1][0] if rows else ''}")

if __name__ == '__main__':
    main()