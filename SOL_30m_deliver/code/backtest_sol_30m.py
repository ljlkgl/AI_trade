# -*- coding: utf-8 -*-
"""SOL 30m 回测：先用 BTC 稳健档(vt=0.10,alpha=0.40,5x)复现，再做小网格确认最优。"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trend_strategy import run

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data','SOLUSDT_30m_kline.csv')
BPD = 48; LEV = 5.0

def show(m):
    cost=m['cost']; net=m['net']
    gross=net+cost
    cr=float(np.sum(cost))/float(np.sum(gross)) if np.sum(gross)!=0 else float('nan')
    print(f"年化 {m['annual_return']*100:8.1f}% | 波动 {m['annual_volatility']*100:5.1f}% | "
          f"夏普 {m['sharpe']:4.2f} | 回撤 {m['max_drawdown']*100:5.1f}% | "
          f"linR2 {m['lin_r2']:.2f} | seg_std {m['seg_std']:.3f} | 杠杆 {m['mean_leverage']:.2f} | "
          f"费占毛利 {cr*100:.1f}% | 样本外 {m['start']}~{m['idx'][-1].date()}")

print("==== SOL 30m · BTC稳健档 vt=0.10 alpha=0.40 5x ====")
m = run(csv=CSV, bars_per_day=BPD, train_months=24, max_leverage=LEV,
        vol_target=0.10, smooth_alpha=0.40, report=False)
show(m)

print("\n==== SOL 30m · 小网格确认 (5x) ====")
print(f"{'vt':>6}{'alpha':>7}{'年化%':>9}{'夏普':>6}{'回撤%':>7}{'linR2':>7}{'杠杆':>6}{'费占%':>7}")
for vt, a in itertools.product([0.06,0.08,0.10,0.13,0.16], [0.30,0.40,0.50]):
    mm = run(csv=CSV, bars_per_day=BPD, train_months=24, max_leverage=LEV,
             vol_target=vt, smooth_alpha=a, report=False)
    cost=mm['cost']; net=mm['net']
    cr=float(np.sum(cost))/float(np.sum(net+cost)) if np.sum(net+cost)!=0 else float('nan')
    print(f"{vt:6.2f}{a:7.2f}{mm['annual_return']*100:9.1f}{mm['sharpe']:6.2f}"
          f"{mm['max_drawdown']*100:7.1f}{mm['lin_r2']:7.2f}{mm['mean_leverage']:6.2f}{cr*100:7.1f}")