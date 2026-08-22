# -*- coding: utf-8 -*-
"""保存 SOL 30m 5x 稳健版(低回撤档 vt=0.06, alpha=0.50) 到 versions/SOL_30m/。
含 config/metrics/oos_backtest.npz/equity.png，手续费明细一并写入 metrics。
"""
import os, sys, json
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

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV=os.path.join(ROOT,'data','SOLUSDT_30m_kline.csv')
BPD=48; PPY=365*BPD
CONF=dict(train_months=24, max_leverage=5.0, vol_target=0.10, smooth_alpha=0.40)
VROOT=os.path.join(ROOT,'versions','SOL_30m','v1_robust')
os.makedirs(VROOT,exist_ok=True)

m=run(csv=CSV,bars_per_day=BPD,**CONF,report=True)

metrics={k:round(float(m[k]),4) for k in
         ['annual_return','annual_volatility','sharpe','max_drawdown','seg_std','autocorr','lin_r2','mean_leverage']}
metrics['start']=m['start']; metrics['end']=str(m['idx'][-1].date())
net=m['net']; cost=m['cost']; pos=m['pos']
gross=net+cost
metrics['total_turnover']=round(float(np.sum(np.abs(np.diff(np.concatenate([[0],pos]))))),2)
metrics['total_cost']=round(float(np.sum(cost)),6)
metrics['cost_vs_gross']=round(float(np.sum(cost))/float(np.sum(gross)),4)
metrics['fee_per_year']=round(float(np.sum(cost))/ (len(net)/PPY),6)
metrics['fee_rate']=0.0004; metrics['slip_rate']=0.0001

json.dump({**CONF,'bars_per_day':BPD,'fee_rate':0.0004,'slip_rate':0.0001},
          open(os.path.join(VROOT,'config.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(metrics,open(os.path.join(VROOT,'metrics.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)

eq=m['eq']; dd=-(eq-np.maximum.accumulate(eq))
np.savez(os.path.join(VROOT,'oos_backtest.npz'),
         dates=np.asarray(m['idx'],dtype='datetime64[D]'),
         close=m['close'],pos=pos,net=net,cost=cost,gross=gross,eq=eq,drawdown=dd)

fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[3,1,1]})
a1,a2,a3=axes
a1.plot(m['idx'],np.exp(eq),color='#1FA28F',lw=1.0); a1.set_yscale('log')
a1.set_title(f"SOL 30m v1稳健版 年化{m['annual_return']*100:.0f}% 回撤{m['max_drawdown']*100:.0f}% "
             f"夏普{m['sharpe']:.2f} linR2={m['lin_r2']:.2f}"); a1.grid(alpha=0.3)
a2.fill_between(m['idx'],0,dd*100,color='#E8463A',alpha=0.5); a2.set_ylabel('回撤%'); a2.grid(alpha=0.3)
a3.plot(m['idx'],pos,color='#22A5F7',lw=0.3); a3.axhline(5,color='#E8463A',lw=0.8,ls='--'); a3.axhline(-5,color='#E8463A',lw=0.8,ls='--')
a3.set_ylabel('持仓(±5)'); a3.set_xlabel('时间'); a3.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VROOT,'equity.png'),dpi=110); plt.close(fig)

print(f"[已保存] {VROOT}")
print(f"手续费: 总换手={metrics['total_turnover']} 总费={metrics['total_cost']} "
      f"占毛利={metrics['cost_vs_gross']*100:.1f}% 年均={metrics['fee_per_year']:.3f}")