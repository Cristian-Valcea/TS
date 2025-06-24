# trading/ib_order_tracker.py
import os
import sys
import asyncio
from ib_insync import Trade
from datetime import datetime

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from Utils.WebsocketUtils import post_context_update
from TraderAgent.utils.db import init_db,insert_trade_log,insert_fill

def log_trade_event(prefix: str, trade: Trade):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    contract = trade.contract.symbol
    order = trade.order
    status = trade.orderStatus

    print(f"[{ts}] {prefix} | Symbol: {contract} | "
          f"Action: {order.action} | Qty: {order.totalQuantity} | "
          f"Status: {status.status} | Filled: {status.filled} | Avg Price: {status.avgFillPrice}")

    # Save to DB
    insert_trade_log(
        symbol=contract,
        action=order.action,
        quantity=order.totalQuantity,
        status=status.status,
        filled=status.filled,
        avg_price=status.avgFillPrice
        )

def register_trade_event_handlers( trade: Trade):
    def on_status_change(trade: Trade):
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        avg_price = trade.orderStatus.avgFillPrice

        info = {
            "symbol": trade.contract.symbol,
            "action": trade.order.action,
            "quantity": trade.order.totalQuantity,
            "status": status,
            "filled": filled,
            "avg_price": avg_price
        }
        log_trade_event(f"📘 ORDER STATUS UPDATE | Symbol: {info['symbol']} | Action: {info['action']} | Qty: {info['quantity']} | Status: {status} | Filled: {filled} | Avg Price: {avg_price}", trade)

        # 🔄 POST STATUS TO CONTEXT
        asyncio.create_task(post_context_update("trader-agent", {
            "latest_order": info,
            "timestamp": datetime.utcnow().isoformat()
        }))


    def on_filled(trade_filled: Trade, fill):
        log_trade_event("✅ ORDER FILLED", trade_filled)
        # 1. Persist to DB
        insert_fill(
            symbol=trade.contract.symbol,
            action=trade.order.action,
            quantity=fill.execution.shares,
            price=fill.execution.price,
            timestamp=fill.execution.time.isoformat()
        )
        # 2. Emit to dashboard
        asyncio.create_task(
            post_context_update("trader-agent",{
            "task": "order_filled",
            "payload": {
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": fill.execution.shares,
                "price": fill.execution.price,
                "timestamp": fill.execution.time.isoformat()
            },
            "from": "trader-agent",
            "status": "filled",
            "timestamp": datetime.utcnow().isoformat()
        })
        )


    trade.fillEvent += on_filled
    trade.statusEvent += on_status_change

