"""HTTP 状态服务器：在浏览器中展示当前交易情况。

用法：
  python web.py                 # 默认端口 8080
  python web.py --port 9000     # 自定义端口
  python web.py --host 0.0.0.0  # 允许局域网访问

页面数据来源：
- 实时部分（账户/持仓/未成交挂单/当前价）：从币安 API 拉取（读取 .env 凭证）；
  拉取失败时降级为展示最近一轮记录中的账户快照。
- 本地状态（操作理由列表/经验库/唤醒条件/轮次历史）：读取 state/ 下的 JSON 文件。

仅用 Python 标准库，无额外依赖。
"""
from __future__ import annotations

import argparse
import html
import importlib
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from config import config
from trading.binance_client import BinanceClient

logger = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).resolve().parent / "state"
# 不把完整挂单列表塞进 JSON API 太长时，轮次记录展示数量上限
_ROUNDS_SHOW = 20


def _read_json(name: str) -> Any:
    """读取 state 下 JSON 文件，失败/缺失返回空容器。"""
    path = _STATE_DIR / name
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取 %s 失败: %s", name, exc)
        return []


def _merge_open_orders(symbol: str, client: BinanceClient) -> list[dict]:
    """合并某币种普通订单与算法条件单（止盈止损），统一为普通订单字段形状。"""
    orders: list[dict] = []
    try:
        orders = list(client.get_open_orders(symbol))
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取 %s 普通挂单失败: %s", symbol, exc)
    try:
        for ao in client.get_open_algo_orders(symbol):
            orders.append({
                "orderId": ao.get("algoId"),
                "symbol": ao.get("symbol"),
                "side": ao.get("side"),
                "type": ao.get("orderType") or ao.get("type"),
                "price": ao.get("price"),
                "stopPrice": ao.get("triggerPrice"),
                "origQty": ao.get("quantity"),
                "status": ao.get("algoStatus") or ao.get("status"),
                "is_algo": True,
            })
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取 %s 条件单(Algo)失败: %s", symbol, exc)
    return orders


class StatusCollector:
    """汇总页面所需数据：实时账户 + 本地状态文件。"""

    def __init__(self) -> None:
        self._client: Optional[BinanceClient] = None

    @property
    def client(self) -> Optional[BinanceClient]:
        if self._client is None and config.binance_api_key and config.binance_api_secret:
            try:
                self._client = BinanceClient(
                    api_key=config.binance_api_key,
                    api_secret=config.binance_api_secret,
                    testnet=config.binance_testnet,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("初始化币安客户端失败: %s", exc)
                self._client = None
        return self._client

    def collect(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "now": _now_str(),
            "testnet": config.binance_testnet,
            "dry_run": config.dry_run,
            "symbols": config.symbols,
            "live": False,  # 实时账户是否拉取成功
        }
        live = self.client
        if live is None:
            return self._fill_from_rounds(status)

        # 实时拉取放入守护线程并限时：币安接口慢/失败时页面仍能快速返回（降级为快照）
        live_data: dict[str, Any] = {}
        done = threading.Event()

        def _work() -> None:
            try:
                prices: dict[str, float] = {}
                for sym in config.symbols:
                    try:
                        prices[sym] = live.get_ticker_price(sym)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("获取 %s 价格失败: %s", sym, exc)
                live_data["prices"] = prices

                account = live.get_account()
                live_data["account"] = {
                    "margin_balance": account.margin_balance,
                    "available_balance": account.available_balance,
                    "unrealized_pnl": account.unrealized_pnl,
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "side": "LONG" if p.position_amt > 0 else "SHORT",
                            "qty": abs(p.position_amt),
                            "entry": p.entry_price,
                            "mark": p.mark_price,
                            "pnl": p.unrealized_pnl,
                            "leverage": p.leverage,
                            "liq": p.liquidation_price,
                        }
                        for p in account.positions
                    ],
                }
                live_data["live"] = True

                live_data["open_orders"] = {
                    sym: _merge_open_orders(sym, live) for sym in config.symbols
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("实时账户拉取异常: %s", exc)
            finally:
                done.set()

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            # 超时：放弃实时数据，快速降级为最近一轮快照
            logger.warning("实时账户拉取超时，状态面板使用最近一轮快照")
            return self._fill_from_rounds(status)

        status.update(live_data)

        # 本地状态
        status["theses"] = _read_json("theses.json")
        status["experiences"] = _read_json("experience_library.json")
        status["watch"] = _read_json("watch_triggers.json")
        rounds = _read_json("rounds.json")
        status["rounds"] = list(reversed(rounds))[:_ROUNDS_SHOW]
        return status

    def _fill_from_rounds(self, status: dict) -> dict[str, Any]:
        """币安不可用：从最近一轮记录取账户/挂单快照展示。"""
        rounds = _read_json("rounds.json")
        status["rounds"] = list(reversed(rounds))[:_ROUNDS_SHOW]
        latest = rounds[-1] if rounds else {}
        status["account"] = latest.get("account", {})
        status["open_orders"] = latest.get("open_orders", {})
        status["prices"] = {}
        status["theses"] = _read_json("theses.json")
        status["experiences"] = _read_json("experience_library.json")
        status["watch"] = _read_json("watch_triggers.json")
        status["note"] = "实时账户拉取失败（检查 API 配置/网络），以下账户与挂单为最近一轮快照"
        return status


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------- HTML 渲染 ----------------

def _fmt(v: Any, digits: int = 2) -> str:
    """数值格式化，None/非数值原样显示。"""
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        try:
            return f"{float(v):,.{digits}f}"
        except (ValueError, TypeError):
            return str(v)
    return html.escape(str(v))


def _render(status: dict[str, Any]) -> str:
    h = html.escape
    rows: list[str] = []
    # 预定义带引号的 HTML 标签片段，避免在 f-string 表达式内使用反斜杠
    # （Python < 3.12 的 f-string 表达式不允许反斜杠）
    algo_tag = '<span class="tag algo">algo</span>'
    error_tag = ' <span class="tag neg">error</span> '

    # ---- 头部 ----
    rows.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>交易系统状态</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background: #0f1420; color: #dfe6f0; }}
  header {{ padding: 16px 24px; background: #161d2e; border-bottom: 1px solid #243049; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .meta {{ margin-top: 6px; font-size: 13px; color: #8ba0bd; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; padding: 20px 24px; }}
  .card {{ background: #161d2e; border: 1px solid #243049; border-radius: 10px; padding: 14px 16px; }}
  .card h2 {{ margin: 0 0 10px; font-size: 15px; color: #7cc4ff; border-left: 3px solid #2f81f7; padding-left: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #1f2a40; }}
  th {{ color: #8ba0bd; font-weight: 500; }}
  .pos {{ color: #3ddc84; }} .neg {{ color: #ff5c6c; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; }}
  .tag.long {{ background: #123524; color: #3ddc84; }} .tag.short {{ background: #3a1620; color: #ff5c6c; }}
  .tag.algo {{ background: #2a2a12; color: #e0c34c; }}
  .muted {{ color: #6b7f9c; font-size: 12px; }}
  .wrap {{ word-break: break-all; white-space: pre-wrap; }}
  .child {{ padding-left: 22px !important; color: #a8b8cf; }}
  ul {{ margin: 4px 0; padding-left: 18px; }}
  li {{ margin: 3px 0; }}
</style>
</head>
<body>
<header>
  <h1>币安永续合约 AI 交易系统 · 状态面板</h1>
  <div class="meta">
    时间: {h(_fmt(status.get('now')))} ·
    测试网: {'是' if status.get('testnet') else '否'} ·
    DRY_RUN: {'是' if status.get('dry_run') else '否'} ·
    标的: {h(', '.join(status.get('symbols', [])))}
    {' · <span class="muted">' + h(status['note']) + '</span>' if status.get('note') else ''}
  </div>
</header>
<div class="grid">""")

    # ---- 账户概况 ----
    acc = status.get("account") or {}
    positions = acc.get("positions") or []
    rows.append(f"""<div class="card"><h2>账户概况</h2>
<table>
<tr><th>权益(margin_balance)</th><td>{_fmt(acc.get('margin_balance'))} USDT</td></tr>
<tr><th>可用余额</th><td>{_fmt(acc.get('available_balance'))} USDT</td></tr>
<tr><th>未实现盈亏</th><td class="{('pos' if (acc.get('unrealized_pnl') or 0) >= 0 else 'neg')}">{_fmt(acc.get('unrealized_pnl'))} USDT</td></tr>
<tr><th>持仓数量</th><td>{len(positions)}</td></tr>
</table></div>""")

    # ---- 持仓明细 ----
    if positions:
        pr = ["<table><tr><th>币种</th><th>方向</th><th>数量</th><th>开仓价</th><th>标记价</th><th>未实现盈亏</th><th>杠杆</th><th>强平价</th></tr>"]
        for p in positions:
            side = "LONG" if (p.get("side") == "LONG") else "SHORT"
            cls = "pos" if side == "LONG" else "neg"
            pnl = p.get("pnl") or 0
            pnl_cls = "pos" if pnl >= 0 else "neg"
            pr.append(
                f"<tr><td>{h(_fmt(p.get('symbol')))}</td>"
                f"<td><span class='tag {cls.lower()}'>{side}</span></td>"
                f"<td>{_fmt(p.get('qty'), 4)}</td>"
                f"<td>{_fmt(p.get('entry'), 6)}</td>"
                f"<td>{_fmt(p.get('mark'), 6)}</td>"
                f"<td class='{pnl_cls}'>{_fmt(pnl)}</td>"
                f"<td>{_fmt(p.get('leverage'), 0)}x</td>"
                f"<td>{_fmt(p.get('liq'), 6)}</td></tr>"
            )
        pr.append("</table>")
        rows.append(f"<div class='card'><h2>持仓明细</h2>{''.join(pr)}</div>")
    else:
        rows.append("<div class='card'><h2>持仓明细</h2><p class='muted'>当前无持仓</p></div>")

    # ---- 未成交挂单 ----
    oo = status.get("open_orders") or {}
    flat = [o for orders in oo.values() for o in orders]
    if flat:
        orows = ["<table><tr><th>币种</th><th>方向</th><th>类型</th><th>价格</th><th>触发价</th><th>数量</th><th>状态</th></tr>"]
        for o in flat:
            side = o.get("side", "-")
            side_cls = "pos" if str(side).upper() == "BUY" else "neg"
            is_algo = o.get("is_algo")
            orows.append(
                f"<tr><td>{h(_fmt(o.get('symbol')))}</td>"
                f"<td class='{side_cls}'>{h(side)}</td>"
                f"<td>{h(_fmt(o.get('type')))}"
                f"{algo_tag if is_algo else ''}</td>"
                f"<td>{_fmt(o.get('price'), 6)}</td>"
                f"<td>{_fmt(o.get('stopPrice'), 6)}</td>"
                f"<td>{_fmt(o.get('origQty'), 4)}</td>"
                f"<td>{h(_fmt(o.get('status')))}</td></tr>"
            )
        orows.append("</table>")
        rows.append(f"<div class='card'><h2>未成交挂单</h2>{''.join(orows)}</div>")
    else:
        rows.append("<div class='card'><h2>未成交挂单</h2><p class='muted'>当前无未成交挂单</p></div>")

    # ---- 当前价格 ----
    prices = status.get("prices") or {}
    if prices:
        prows = ["<table><tr><th>币种</th><th>现价</th></tr>"]
        for sym, px in prices.items():
            prows.append(f"<tr><td>{h(sym)}</td><td>{_fmt(px, 6)}</td></tr>")
        prows.append("</table>")
        rows.append(f"<div class='card'><h2>当前价格</h2>{''.join(prows)}</div>")

    # ---- 操作理由列表 ----
    theses = status.get("theses") or []
    rows.append(f"<div class='card'><h2>操作理由列表（{len(theses)} 条）</h2>")
    if theses:
        by_id = {t.get("id"): t for t in theses}
        # 父子层级：先顶层，再递归展示子节点
        def _depth(t: dict) -> int:
            d = 0
            pid = t.get("parent_id")
            seen: set[str] = set()
            while pid and pid in by_id and pid not in seen:
                seen.add(pid)
                d += 1
                pid = by_id[pid].get("parent_id")
            return d
        srows = ["<table><tr><th>编号</th><th>类型</th><th>方向</th><th>价格</th><th>理由</th></tr>"]
        for t in sorted(theses, key=lambda x: (x.get("parent_id") or "", x.get("id") or "")):
            d = _depth(t)
            kind = t.get("kind", "position")
            direction = t.get("direction") or ""
            dr_cls = "pos" if direction == "LONG" else ("neg" if direction == "SHORT" else "")
            srows.append(
                f"<tr class='{'child' if d else ''}'>"
                f"<td>{h(_fmt(t.get('id')))}</td>"
                f"<td>{h(kind)}</td>"
                f"<td class='{dr_cls}'>{h(direction)}</td>"
                f"<td>{_fmt(t.get('entry_price'), 6)}</td>"
                f"<td class='wrap'>{h(_fmt(t.get('thesis')))}</td></tr>"
            )
        srows.append("</table>")
        rows.append("".join(srows))
    else:
        rows.append("<p class='muted'>当前无进行中操作理由</p>")
    rows.append("</div>")

    # ---- 唤醒条件 ----
    watch = status.get("watch") or []
    rows.append(f"<div class='card'><h2>唤醒条件（{len(watch)} 个）</h2>")
    if watch:
        wrows = ["<table><tr><th>币种</th><th>条件</th><th>值</th><th>过期时间</th></tr>"]
        for w in watch:
            wrows.append(
                f"<tr><td>{h(_fmt(w.get('symbol')))}</td>"
                f"<td>{h(_fmt(w.get('condition')))}</td>"
                f"<td>{_fmt(w.get('value'), 6)}</td>"
                f"<td>{h(_fmt(w.get('expires_at')))}</td></tr>"
            )
        wrows.append("</table>")
        rows.append("".join(wrows))
    else:
        rows.append("<p class='muted'>无唤醒条件</p>")
    rows.append("</div>")

    # ---- 最近轮次 ----
    rounds = status.get("rounds") or []
    rows.append(f"<div class='card'><h2>最近轮次（最近 {len(rounds)} 轮）</h2>")
    if not rounds:
        rows.append("<p class='muted'>暂无轮次记录</p>")
    for r in rounds:
        ts = r.get("timestamp", "-")
        exec_list = r.get("execution") or []
        inst_list = r.get("instructions_after_risk") or []
        blocked = r.get("risk_blocked") or []
        rows.append(f"<div style='margin-bottom:12px'>")
        rows.append(
            f"<div><b>{h(ts)}</b>"
            f"{error_tag + h(str(r.get('error'))) if r.get('error') else ''}"
            f"</div>"
        )
        ma = r.get("market_assessment")
        if ma:
            rows.append(f"<div class='muted wrap'>市场评估: {h(str(ma)[:400])}</div>")
        if blocked:
            rows.append(f"<div class='muted'>被风控拦截: {h('; '.join(str(b.get('errors')) for b in blocked))}</div>")
        if inst_list:
            rows.append("<div class='muted'>指令:</div><ul>")
            for ins in inst_list:
                rows.append(
                    f"<li>{h(_fmt(ins.get('symbol')))} {h(_fmt(ins.get('action')))} "
                    f"type={h(_fmt(ins.get('order_type')))} "
                    f"price={_fmt(ins.get('price'), 6)} "
                    f"margin={_fmt(ins.get('margin'))} lev={_fmt(ins.get('leverage'), 0)}x</li>"
                )
            rows.append("</ul>")
        if exec_list:
            rows.append("<div class='muted'>执行结果:</div><ul>")
            for e in exec_list:
                st = e.get("status")
                st_cls = "pos" if st in ("OPENED", "DRY_RUN", "CLOSED") else ("neg" if st in ("FAILED", "SKIPPED") else "")
                rows.append(
                    f"<li>{h(_fmt(e.get('symbol')))} {h(_fmt(e.get('action')))} "
                    f"-> <span class='{st_cls}'>{h(_fmt(st))}</span>"
                    f"{' ' + h(str(e.get('error'))) if e.get('error') else ''}</li>"
                )
            rows.append("</ul>")
        rows.append("</div>")
    rows.append("</div>")

    rows.append("</div></body></html>")
    return "\n".join(rows)


class Handler(BaseHTTPRequestHandler):
    collector: StatusCollector  # 类级共享

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        # 访问密码校验：URL 以 ?password=xxx 传递，错误/缺失则无响应直接断开，防止被探测
        if parse_qs(parsed.query).get("password", [""])[0] != config.web_password:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass
            return
        path = parsed.path
        if path in ("/", "/index.html"):
            try:
                status = self.collector.collect()
            except Exception as exc:  # noqa: BLE001
                logger.exception("收集状态失败: %s", exc)
                self._send(500, "text/plain; charset=utf-8", f"状态收集失败: {exc}".encode("utf-8"))
                return
            body = _render(status).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/api/status":
            try:
                status = self.collector.collect()
            except Exception as exc:  # noqa: BLE001
                logger.exception("收集状态失败: %s", exc)
                self._send(500, "application/json", json.dumps({"error": str(exc)}).encode("utf-8"))
                return
            body = json.dumps(status, ensure_ascii=False, default=str).encode("utf-8")
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not Found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # 请求行形如 `GET /?password=xxx HTTP/1.1`，丢弃 query 避免访问密码进入日志
        msg = fmt % args
        parts = msg.split(" ", 2)
        if len(parts) == 3 and "?" in parts[1]:
            msg = f"{parts[0]} {parts[1].split('?', 1)[0]} {parts[2]}"
        logger.info("web: %s - %s", self.address_string(), msg)


def start_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """启动 HTTP 服务器（非阻塞，需调用方自行 serve_forever 或另起线程）。"""
    collector = StatusCollector()
    Handler.collector = collector
    return ThreadingHTTPServer((host, port), Handler)


class WebServerController:
    """管理状态面板服务器的生命周期，供主进程内嵌调用。

    - 随时终止：监听控制文件 state/web_ctl.json，写入 {"cmd":"stop"} 即停止
      （"start" 重新启动，"restart" 重载并重启），无需重启主进程；
    - 热更新：周期检测 web.py 文件修改时间，变化后自动 importlib.reload
      并重启服务器，主进程运行中修改 web.py 保存即可立即生效。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctl_path: Optional[Path] = None,
        web_path: Optional[Path] = None,
        check_interval: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.ctl_path = ctl_path or (_STATE_DIR / "web_ctl.json")
        self.web_path = web_path or (Path(__file__).resolve())
        self.check_interval = check_interval
        self._server: Optional[ThreadingHTTPServer] = None
        self._mtime = self._file_mtime(self.web_path)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """启动监听线程（守护线程，不影响主进程交易循环）。"""
        if self._running:
            return
        self._running = True
        self._ensure_server()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """终止监听线程并关闭服务器。"""
        self._running = False
        self._shutdown_server()

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._apply_control(self._read_ctl())
            except Exception as exc:  # noqa: BLE001
                logger.warning("处理 web 控制命令失败: %s", exc)
            try:
                self._check_reload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("web 热更新检测失败: %s", exc)

    # ---------- 服务器管理 ----------

    def _ensure_server(self) -> None:
        if self._server is not None:
            return
        server = start_server(self.host, self.port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._server = server
        logger.info("状态面板已启动: http://%s:%d", self.host, self.port)

    def _shutdown_server(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭状态面板失败: %s", exc)
        self._server = None
        logger.info("状态面板已停止")

    # ---------- 外部控制（控制文件） ----------

    def _read_ctl(self) -> Optional[str]:
        if not self.ctl_path.exists():
            return None
        try:
            with open(self.ctl_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("cmd") if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def _clear_ctl(self) -> None:
        try:
            self.ctl_path.write_text('{"cmd": null}', encoding="utf-8")
        except OSError as exc:
            logger.warning("清除 web 控制命令失败: %s", exc)

    def _apply_control(self, cmd: Optional[str]) -> None:
        if cmd not in ("stop", "start", "restart"):
            return
        logger.info("收到状态面板控制命令: %s", cmd)
        if cmd == "stop":
            self._shutdown_server()
        elif cmd == "start":
            self._ensure_server()
        elif cmd == "restart":
            self._reload_and_restart()
        self._clear_ctl()

    # ---------- 热更新（web.py 代码变更自动 reload） ----------

    @staticmethod
    def _file_mtime(path: Path) -> Optional[int]:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def _check_reload(self) -> None:
        mtime = self._file_mtime(self.web_path)
        if mtime is None or mtime == self._mtime:
            return
        logger.info("检测到 web.py 已更新，自动重载并重启状态面板...")
        self._reload_and_restart()
        self._mtime = mtime

    def _reload_and_restart(self) -> None:
        importlib.reload(importlib.import_module("web"))
        self._shutdown_server()
        self._ensure_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="交易系统状态 HTTP 服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = start_server(args.host, args.port)
    print(f"状态面板已启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
