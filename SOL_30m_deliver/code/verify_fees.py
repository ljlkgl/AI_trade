# -*- coding: utf-8 -*-
"""对 v1 存档做逐年手续费/换手/收益明细，并输出图表，交叉验证数字真实性。"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for _f in ['Microsoft YaHei', 'PingFang SC', 'SimHei']:
    matplotlib.rcParams['font.sans-serif'] = [_f]
    try:
        plt.rcParams['font.family'] = _f
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEE_RATE = 0.0004
SLIP_RATE = 0.0001
TOTAL = FEE_RATE + SLIP_RATE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS = ['ETH', 'BTC']


def load(asset):
    vdir = os.path.join(ROOT, 'versions', asset, 'v1_trend86')
    d = np.load(os.path.join(vdir, 'oos_backtest.npz'), allow_pickle=True)
    dates = pd.to_datetime(d['dates'])
    pos = d['pos']
    net = d['net']          # 已扣费的净日收益(对数)
    cost = d['cost']        # 每日调仓成本(对数)
    close = d['close']
    dd = d['drawdown']
    eq = d['eq']
    # 毛利(re-run 口径): net + cost
    gross = net + cost
    return dates, close, pos, net, cost, gross, eq, dd


def yearly_stats(dates, net, cost, gross, pos):
    df = pd.DataFrame({'date': dates, 'net': net, 'cost': cost,
                       'gross': gross, 'turn': np.abs(np.diff(np.concatenate([[0], pos])))})
    df['year'] = df['date'].dt.year
    rows = []
    for y, g in df.groupby('year'):
        sum_net = float(g['net'].sum())
        sum_cost = float(g['cost'].sum())
        sum_gross = float(g['gross'].sum())
        rows.append(dict(
            year=int(y), days=len(g),
            net_ret=sum_net,              # 净收益(对数累计)
            gross_ret=sum_gross,          # 毛利
            cost=sum_cost,                # 手续费
            annualized_net=sum_net,
            turnover=float(g['turn'].sum()),
        ))
    return rows


out = {}
for asset in ASSETS:
    dates, close, pos, net, cost, gross, eq, dd = load(asset)
    rows = yearly_stats(dates, net, cost, gross, pos)
    out[asset] = dict(dates=dates, close=close, pos=pos, net=net, cost=cost,
                      gross=gross, eq=eq, dd=dd, rows=rows)

    # ---- 控制台逐年明细 ----
    print(f"\n===== {asset} 逐年明细 (单边费率 0.04%+0.01%=0.05%) =====")
    print(f"{'年':>5} {'净累计':>8} {'毛利累计':>8} {'手续费':>8} {'费占毛利%':>8} {'换手量':>7}")
    cum_net = cum_g = cum_cost = 0.0
    for r in rows:
        cum_net += r['net_ret']; cum_g += r['gross_ret']; cum_cost += r['cost']
        fg = r['cost'] / r['gross_ret'] * 100 if r['gross_ret'] != 0 else float('nan')
        print(f"{r['year']:>5} {cum_net:8.4f} {cum_g:8.4f} {cum_cost:8.4f} {fg:8.1f} {r['turnover']:7.1f}")
    total_gross = sum(r['gross_ret'] for r in rows)
    total_cost = sum(r['cost'] for r in rows)
    total_net = sum(r['net_ret'] for r in rows)
    print(f"合计: 净{total_net:.4f} 毛利{total_gross:.4f} 手续费{total_cost:.4f} "
          f"费占毛利{total_cost/total_gross if total_gross else 0:.1%}")

    # ---- 图1: 累计净/毛/费 三条线 ----
    cg = np.cumsum(gross); cn = np.cumsum(net); cc = np.cumsum(cost)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, np.exp(cg), label='毛利净值', lw=1.5, color='#7A4DEC')
    ax.plot(dates, np.exp(cn), label='净净值(扣费)', lw=1.5, color='#1FA28F')
    ax.set_yscale('log')
    ax.set_title(f'{asset} · 毛利 vs 净净值(对数)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(ROOT, 'results', f'fee_verify_{asset}_net.png')
    fig.savefig(p1, dpi=110); plt.close(fig)

    # ---- 图2: 逐年费占毛利 与 换手 条形 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    yrs = [r['year'] for r in rows]
    fee_pct = [r['cost'] / r['gross_ret'] * 100 if r['gross_ret'] != 0 else 0 for r in rows]
    ax = axes[0]
    ax.bar(yrs, fee_pct, color='#E8463A', alpha=0.8)
    ax.set_ylabel('手续费/毛利 %'); ax.set_title(f'{asset} · 逐年手续费占比')
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(yrs, [r['turnover'] for r in rows], color='#22A5F7', alpha=0.8)
    ax.set_ylabel('换手量'); ax.set_xlabel('年份')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(ROOT, 'results', f'fee_verify_{asset}_bar.png')
    fig.savefig(p2, dpi=110); plt.close(fig)
    print(f"[已保存] {p1} / {p2}")

# ---- 汇总 json ----
agg = {}
for asset in ASSETS:
    rows = out[asset]['rows']
    agg[asset] = dict(
        total_net=sum(r['net_ret'] for r in rows),
        total_gross=sum(r['gross_ret'] for r in rows),
        total_cost=sum(r['cost'] for r in rows),
        total_turnover=sum(r['turnover'] for r in rows),
    )
with open(os.path.join(ROOT, 'results', 'fee_verify.json'), 'w', encoding='utf-8') as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)
print(f"\n[已保存] {os.path.join(ROOT, 'results', 'fee_verify.json')}")