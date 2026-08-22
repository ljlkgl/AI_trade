# -*- coding: utf-8 -*-
"""SOL 30m 净值图：稳健档(vt=0.10,a=0.40) 与 低回撤档(vt=0.06,a=0.50) 对比。"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for _f in ['Microsoft YaHei','PingFang SC','SimHei']:
    matplotlib.rcParams['font.sans-serif']=[_f]
    try: plt.rcParams['font.family']=_f
    except Exception: continue
matplotlib.rcParams['axes.unicode_minus']=False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trend_strategy import run

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data','SOLUSDT_30m_kline.csv')
BPD=48; LEV=5.0
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT,'results'),exist_ok=True)

def chart(m, fname, title):
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[3,1,1]})
    a1,a2,a3=axes
    a1.plot(m['idx'],np.exp(m['eq']),color='#1FA28F',lw=1.0); a1.set_yscale('log')
    a1.set_title(title); a1.grid(alpha=0.3)
    dd=-(m['eq']-np.maximum.accumulate(m['eq']))
    a2.fill_between(m['idx'],0,dd*100,color='#E8463A',alpha=0.5); a2.set_ylabel('回撤%'); a2.grid(alpha=0.3)
    a3.plot(m['idx'],m['pos'],color='#22A5F7',lw=0.3); a3.axhline(5,color='#E8463A',lw=0.8,ls='--'); a3.axhline(-5,color='#E8463A',lw=0.8,ls='--')
    a3.set_ylabel('持仓(±5)'); a3.set_xlabel('时间'); a3.grid(alpha=0.3)
    fig.tight_layout()
    p=os.path.join(ROOT,'results',fname); fig.savefig(p,dpi=110); plt.close(fig)
    print(f"[已保存] {p}")

m1=run(csv=CSV,bars_per_day=BPD,train_months=24,max_leverage=LEV,vol_target=0.10,smooth_alpha=0.40,report=False)
chart(m1,'sol_30m_5x_robust.png',
      f"SOL 30m 5x稳健档(vt=0.10,a=0.40) 年化{m1['annual_return']*100:.0f}% "
      f"回撤{m1['max_drawdown']*100:.0f}% 夏普{m1['sharpe']:.2f} linR2={m1['lin_r2']:.2f}")

m2=run(csv=CSV,bars_per_day=BPD,train_months=24,max_leverage=LEV,vol_target=0.06,smooth_alpha=0.50,report=False)
chart(m2,'sol_30m_5x_lowdd.png',
      f"SOL 30m 5x低回撤档(vt=0.06,a=0.50) 年化{m2['annual_return']*100:.0f}% "
      f"回撤{m2['max_drawdown']*100:.0f}% 夏普{m2['sharpe']:.2f} linR2={m2['lin_r2']:.2f}")