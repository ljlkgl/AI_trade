#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
固化 v1 基线趋势跟踪版本到 versions/
=======================================
v1 - 基线趋势跟踪（86% 档）：多尺度动量 + 波动率目标 + 轻量平滑，每月滚动重训，
     回测已计入调仓成本（单边手续费 0.04% + 滑点 0.01%，按每日换手量计费）。

每个版本目录输出：
  config.json       - 复现该版本所需的全部参数（含成本费率）
  metrics.json      - 回测指标（含收益/回撤/夏普/平滑度 + 调仓与手续费统计）
  oos_backtest.npz  - dates/close/pos/net/cost/eq/drawdown(样本外全序列)
  equity.png        - 净值/回撤/持仓 三合一图

用法（项目根目录）：python scripts/save_versions.py
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.trend_strategy import run

# 中文字体
for f in ['Microsoft YaHei', 'PingFang SC', 'SimHei']:
    matplotlib.rcParams['font.sans-serif'] = [f]
    try:
        plt.rcParams['font.family'] = f
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

VERSIONS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'versions')

FEE_RATE = 0.0004
SLIP_RATE = 0.0001

ASSETS = [
    dict(key='ETH', csv='data/ETHUSDT_1D_kline.csv',
         conf=dict(train_months=24, max_leverage=1.0, vol_target=0.48, smooth_alpha=0.30),
         label='基线趋势86%'),
    dict(key='BTC', csv='data/BTCUSDT_1D_kline.csv',
         conf=dict(train_months=24, max_leverage=1.0, vol_target=0.37, smooth_alpha=0.40),
         label='基线趋势86%'),
]

METRIC_FIELDS = [
    ('annual_return', '年化收益'),
    ('annual_volatility', '年化波动'),
    ('sharpe', '夏普'),
    ('max_drawdown', '最大回撤'),
    ('seg_std', 'seg_std'),
    ('autocorr', '自相关'),
    ('lin_r2', 'linR2'),
    ('mean_leverage', '平均杠杆'),
    # 调仓与手续费统计
    ('total_turnover', '样本外总换手量'),
    ('total_cost', '样本外总手续费'),
    ('cost_vs_gross', '手续费占毛利比例'),
    ('fee_per_year', '年均手续费'),
]


def _pick(value):
    return value.item() if isinstance(value, np.generic) else value


def build_metrics(m, cost=None, ppy=365, n_oos=None):
    d = {}
    for k, zh in METRIC_FIELDS:
        v = m.get(k)
        d[k] = round(_pick(v), 4) if isinstance(_pick(v), float) else _pick(v)
    # 附加调仓/手续费统计
    if cost is not None:
        total_cost = float(np.sum(cost))
        total_turn = float(np.sum(np.abs(np.diff(np.concatenate([[0], m['pos']])))))
        d['total_turnover'] = round(total_turn, 2)
        d['total_cost'] = round(total_cost, 4)
        d['cost_vs_gross'] = round(total_cost / (total_cost + np.sum(m['net'])), 4)
        d['fee_per_year'] = round(total_cost / (n_oos / ppy), 4)
    d['fee_rate'] = FEE_RATE
    d['slip_rate'] = SLIP_RATE
    d['start'] = m['start']
    d['end'] = str(m['idx'][-1].date())
    return d


def render_equity(m, outpath, title):
    idx, eq, pos = m['idx'], m['eq'], m['pos']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax_eq, ax_dd, ax_pos = axes
    ax_eq.plot(idx, np.exp(eq), color='#1FA28F', lw=1.5)
    ax_eq.set_title(title)
    ax_eq.set_yscale('log')
    ax_eq.grid(alpha=0.3)
    dd = -(eq - np.maximum.accumulate(eq))
    ax_dd.fill_between(idx, 0, dd * 100, color='#E8463A', alpha=0.5)
    ax_dd.set_ylabel('回撤%')
    ax_dd.grid(alpha=0.3)
    ax_pos.plot(idx, pos, color='#22A5F7', lw=0.8)
    ax_pos.axhline(0, color='#9AA0A6', lw=0.8)
    ax_pos.set_ylabel('持仓')
    ax_pos.set_xlabel('时间')
    ax_pos.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(VERSIONS_ROOT, exist_ok=True)

    for asset in ASSETS:
        key = asset['key']
        csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), asset['csv'])
        out_root = os.path.join(VERSIONS_ROOT, key)
        os.makedirs(out_root, exist_ok=True)

        conf = dict(train_months=24)
        conf.update(asset['conf'])
        vdir = os.path.join(out_root, 'v1_trend86')
        os.makedirs(vdir, exist_ok=True)

        m = run(csv=csv, **conf, report=False)

        eq = m['eq']
        dd = -(eq - np.maximum.accumulate(eq))
        n_oos = len(eq)
        metrics = build_metrics(m, cost=m.get('cost'), ppy=365, n_oos=n_oos)

        json.dump(conf, open(os.path.join(vdir, 'config.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        json.dump(metrics, open(os.path.join(vdir, 'metrics.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        np.savez(os.path.join(vdir, 'oos_backtest.npz'),
                 dates=np.asarray(m['idx'], dtype='datetime64[D]'),
                 close=m['close'], pos=m['pos'], net=m['net'],
                 cost=m['cost'] if 'cost' in m else np.zeros_like(eq),
                 eq=eq, drawdown=dd)

        title = (f"{os.path.basename(csv)} · {asset['label']}(对数) "
                 f"年化={m['annual_return']*100:.0f}% 回撤={m['max_drawdown']*100:.0f}% "
                 f"夏普={m['sharpe']:.2f} linR2={m['lin_r2']:.2f}")
        render_equity(m, os.path.join(vdir, 'equity.png'), title)

        print(f"[已保存] {vdir}")
        print(f"  {asset['label']:<10} 年化 {m['annual_return']*100:6.1f}% | "
              f"回撤 {m['max_drawdown']*100:5.1f}% | 夏普 {m['sharpe']:4.2f} | "
              f"seg_std {m['seg_std']:.3f} | linR2 {m['lin_r2']:.3f}")
        if 'cost' in m:
            print(f"  手续费统计: 总换手={metrics['total_turnover']} 总手续费={metrics['total_cost']} "
                  f"占毛利={metrics['cost_vs_gross']*100:.1f}% 年均={metrics['fee_per_year']:.3f}")


if __name__ == '__main__':
    main()