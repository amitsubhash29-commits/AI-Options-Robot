import os
import csv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".") / ".env")

import time
import json
from math import log1p
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from kiteconnect import KiteConnect
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


class KiteReadError(RuntimeError):
    pass


class CodespaceSelfImprovingRobot:
    def __init__(self):
        self.API_KEY = os.environ.get("KITE_API_KEY")
        self.API_SECRET = os.environ.get("KITE_API_SECRET")
        self.ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
        self.kite = KiteConnect(api_key=self.API_KEY, timeout=20)
        self._authenticate()
        self.bot_traded_symbols = set()
        self.active_trade_symbols = set()
        self.active_trade_pnl_baselines = {}
        self.trade_audit_filename = "trade_audit_log.csv"
        self.open_trade_audit = {}
        self._entry_in_progress = False

        self.start_capital = 200000.00
        self.capital = 200000.00
        self.daily_target = 5000.00
        self.daily_max_loss = 1500.00
        self.trade_target = 2500.00
        self.trade_max_loss = 1000.00
        self.log_filename = "performance_log.json"
        self.execution_quantities = {
            "NIFTY": 65,
            "BANKNIFTY": 30,
            "SENSEX": 20,
        }
        self.indices = {
            "NIFTY": {
                "spot_exchange": "NSE",
                "spot_symbol": "NIFTY 50",
                "option_exchange": "NFO",
                "contract_name": "NIFTY",
                "strike_step": 50,
            },
            "BANKNIFTY": {
                "spot_exchange": "NSE",
                "spot_symbol": "NIFTY BANK",
                "option_exchange": "NFO",
                "contract_name": "BANKNIFTY",
                "strike_step": 100,
            },
            "SENSEX": {
                "spot_exchange": "BSE",
                "spot_symbol": "SENSEX",
                "option_exchange": "BFO",
                "contract_name": "SENSEX",
                "strike_step": 100,
            },
        }
        self.index_tokens = {}
        self.option_contracts = {}
        self.oi_snapshots = {name: {} for name in self.indices}
        self.candle_cache = {name: {} for name in self.indices}
        self.last_order_candle = {}
        
        # Adjustable trading filters
        self.rsi_overbought = 70.0
        self.rsi_oversold = 30.0
        
        # Initialize self-learning loop from yesterday's history
        self.run_self_analysis()

    def _authenticate(self):
        if self.API_KEY == "YOUR_API_KEY" or not self.API_KEY:
            raise ValueError("Set KITE_API_KEY before starting the bot.")
        if self.API_SECRET == "YOUR_API_SECRET" or not self.API_SECRET:
            raise ValueError("Set KITE_API_SECRET before starting the bot.")

        print(self.kite.login_url())
        request_token = input("Enter the request_token from the redirected URL: ").strip()
        if not request_token:
            raise ValueError("A Kite request token is required to start trading.")

        session = self.kite.generate_session(
            request_token, api_secret=self.API_SECRET
        )
        self.ACCESS_TOKEN = session["access_token"]
        self.kite.set_access_token(self.ACCESS_TOKEN)

    def _load_market_instruments(self):
        exchange_cache = {}
        for name, config in self.indices.items():
            spot_exchange = config["spot_exchange"]
            if spot_exchange not in exchange_cache:
                exchange_cache[spot_exchange] = self.kite.instruments(spot_exchange)
            spot = next(
                (
                    instrument
                    for instrument in exchange_cache[spot_exchange]
                    if instrument.get("tradingsymbol") == config["spot_symbol"]
                ),
                None,
            )
            if spot is None:
                raise RuntimeError(f"Could not find {name} on {spot_exchange}.")
            self.index_tokens[name] = spot["instrument_token"]

            option_exchange = config["option_exchange"]
            if option_exchange not in exchange_cache:
                exchange_cache[option_exchange] = self.kite.instruments(option_exchange)
            contracts = [
                instrument
                for instrument in exchange_cache[option_exchange]
                if instrument.get("name") == config["contract_name"]
                and instrument.get("instrument_type") in ("CE", "PE")
                and instrument.get("expiry") is not None
                and instrument["expiry"] >= date.today()
            ]
            if not contracts:
                raise RuntimeError(f"No active option contracts found for {name}.")
            nearest_expiry = min(instrument["expiry"] for instrument in contracts)
            self.option_contracts[name] = [
                instrument
                for instrument in contracts
                if instrument["expiry"] == nearest_expiry
            ]

    def _get_option_chain(self, name, spot_price):
        config = self.indices[name]
        strike_step = config["strike_step"]
        atm_strike = round(spot_price / strike_step) * strike_step
        contracts = [
            instrument
            for instrument in self.option_contracts[name]
            if abs(instrument["strike"] - atm_strike) <= 6 * strike_step
        ]
        if not contracts:
            return None

        quote_keys = [
            f"{config['option_exchange']}:{instrument['tradingsymbol']}"
            for instrument in contracts
        ]
        quotes = self.kite.quote(*quote_keys)
        previous = self.oi_snapshots[name]
        chain = {"CE": [], "PE": []}

        for instrument, quote_key in zip(contracts, quote_keys):
            quote = quotes.get(quote_key, {})
            row = {
                **instrument,
                "oi": float(quote.get("oi") or 0),
                "last_price": float(quote.get("last_price") or 0),
                "writing_delta": 0.0,
            }
            old_quote = previous.get(instrument["tradingsymbol"])
            if old_quote and row["oi"] > old_quote["oi"] and row["last_price"] <= old_quote["last_price"]:
                row["writing_delta"] = row["oi"] - old_quote["oi"]
            previous[instrument["tradingsymbol"]] = {
                "oi": row["oi"],
                "last_price": row["last_price"],
            }
            chain[instrument["instrument_type"]].append(row)

        chain["support"] = max(
            (row for row in chain["PE"] if row["strike"] < spot_price),
            key=lambda row: row["oi"],
            default=None,
        )
        chain["resistance"] = max(
            (row for row in chain["CE"] if row["strike"] > spot_price),
            key=lambda row: row["oi"],
            default=None,
        )
        chain["put_writing"] = max(
            (row["writing_delta"] for row in chain["PE"]), default=0.0
        )
        chain["call_writing"] = max(
            (row["writing_delta"] for row in chain["CE"]), default=0.0
        )
        chain["put_oi"] = max((row["oi"] for row in chain["PE"]), default=0.0)
        chain["call_oi"] = max((row["oi"] for row in chain["CE"]), default=0.0)
        return chain

    def _kite_read(self, method_name, *args, **kwargs):
        try:
            return getattr(self.kite, method_name)(*args, **kwargs)
        except Exception as exc:
            print(
                f"⚠️ Kite {method_name} request failed; retrying on the next cycle: {exc}"
            )
            raise KiteReadError(f"Kite {method_name} request failed.") from exc

    def _get_intraday_candles(self, name, interval, now_ist):
        cache = self.candle_cache[name].get(interval)
        today = now_ist.date()
        start = (now_ist - timedelta(days=10)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            candles = self._kite_read(
                "historical_data",
                self.index_tokens[name],
                start,
                now_ist,
                interval=interval,
            )
        except KiteReadError:
            if cache and cache["date"] == today:
                print(f"[CANDLE CACHE] {name} {interval} using this session's cached candles.")
                return cache["candles"]
            print(
                f"[CANDLE CACHE] {name} {interval} has no same-session data; "
                "retrying next cycle."
            )
            return []

        if candles:
            candles = list(candles)
            self.candle_cache[name][interval] = {
                "date": today,
                "candles": candles,
            }
            return candles
        if cache and cache["date"] == today:
            print(f"[CANDLE CACHE] {name} {interval} response was empty; using cached candles.")
            return cache["candles"]
        return []

    @staticmethod
    def _calculate_indicators(candles):
        if len(candles) < 55:
            return None

        frame = pd.DataFrame(candles)
        close = frame["close"].astype(float)
        fast_ema = EMAIndicator(close, window=9).ema_indicator()
        slow_ema = EMAIndicator(close, window=21).ema_indicator()
        trend_ema = EMAIndicator(close, window=50).ema_indicator()
        rsi = RSIIndicator(close, window=14).rsi()
        adx = ADXIndicator(
            frame["high"], frame["low"], close, window=14
        ).adx()
        atr = AverageTrueRange(
            frame["high"], frame["low"], close, window=14
        ).average_true_range()
        return {
            "candles": candles,
            "close": close,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "trend_ema": trend_ema,
            "bullish_cross": (
                fast_ema.iloc[-2] <= slow_ema.iloc[-2]
                and fast_ema.iloc[-1] > slow_ema.iloc[-1]
            ),
            "bearish_cross": (
                fast_ema.iloc[-2] >= slow_ema.iloc[-2]
                and fast_ema.iloc[-1] < slow_ema.iloc[-1]
            ),
            "rsi": float(rsi.iloc[-1]),
            "adx": float(adx.iloc[-1]),
            "atr": float(atr.iloc[-1]),
            "atr_percent": float(atr.iloc[-1]) / float(close.iloc[-1]),
            "last_close": float(close.iloc[-1]),
        }

    def _analyze_index(self, name):
        config = self.indices[name]
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        minute_candles = self._get_intraday_candles(name, "minute", now_ist)
        five_minute_candles = self._get_intraday_candles(
            name, "5minute", now_ist
        )
        minute = self._calculate_indicators(minute_candles)
        five_minute = self._calculate_indicators(five_minute_candles)
        if minute is None or five_minute is None:
            print(
                f"[MARKET DATA] {name} waiting for 55 candles per timeframe "
                f"(1m={len(minute_candles)}, 5m={len(five_minute_candles)})."
            )
            return None

        minute_fast = minute["fast_ema"]
        minute_slow = minute["slow_ema"]
        five_fast = five_minute["fast_ema"]
        five_slow = five_minute["slow_ema"]
        current_rsi = minute["rsi"]
        current_adx = minute["adx"]
        atr_percent = minute["atr_percent"]
        quote_key = f"{config['spot_exchange']}:{config['spot_symbol']}"
        spot_price = float(self._kite_read("ltp", quote_key)[quote_key]["last_price"])
        print(
            f"[MARKET DATA] {name} spot={spot_price:.2f} "
            f"1m RSI={minute['rsi']:.1f} ADX={minute['adx']:.1f} "
            f"ATR%={minute['atr_percent']:.3%} | "
            f"5m RSI={five_minute['rsi']:.1f} ADX={five_minute['adx']:.1f} "
            f"ATR%={five_minute['atr_percent']:.3%} "
            f"EMA-cross 1m={'up' if minute['bullish_cross'] else 'down' if minute['bearish_cross'] else 'none'} "
            f"5m={'up' if five_minute['bullish_cross'] else 'down' if five_minute['bearish_cross'] else 'none'}"
        )
        chain = self._get_option_chain(name, spot_price)
        if chain is None:
            return None

        bullish_momentum = (
            minute_fast.iloc[-1] > minute_slow.iloc[-1]
            and minute["last_close"] > minute["trend_ema"].iloc[-1]
            and (
                minute["bullish_cross"]
                or current_rsi >= 60
                or current_rsi >= self.rsi_overbought
            )
        )
        bearish_momentum = (
            minute_fast.iloc[-1] < minute_slow.iloc[-1]
            and minute["last_close"] < minute["trend_ema"].iloc[-1]
            and (
                minute["bearish_cross"]
                or current_rsi <= 40
                or current_rsi <= self.rsi_oversold
            )
        )
        five_minute_bullish = (
            five_fast.iloc[-1] > five_slow.iloc[-1]
            and five_minute["last_close"] > five_minute["trend_ema"].iloc[-1]
        )
        five_minute_bearish = (
            five_fast.iloc[-1] < five_slow.iloc[-1]
            and five_minute["last_close"] < five_minute["trend_ema"].iloc[-1]
        )
        bullish = bullish_momentum and five_minute_bullish
        bearish = bearish_momentum and five_minute_bearish

        if bullish or bearish:
            if (
                five_minute["adx"] < 18
                and five_minute["atr_percent"] < 0.004
            ):
                if bullish and chain["put_writing"] > 0 and chain["support"]:
                    return {
                        "index": name,
                        "mode": "PUT_CREDIT",
                        "score": 2.0 + log1p(chain["put_writing"]),
                        "spot": spot_price,
                        "chain": chain,
                        "candle": minute_candles[-1]["date"],
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr_percent": atr_percent,
                    }
                if bearish and chain["call_writing"] > 0 and chain["resistance"]:
                    return {
                        "index": name,
                        "mode": "CALL_CREDIT",
                        "score": 2.0 + log1p(chain["call_writing"]),
                        "spot": spot_price,
                        "chain": chain,
                        "candle": minute_candles[-1]["date"],
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr_percent": atr_percent,
                    }

            direction = "CE" if bullish else "PE"
            writing = chain["put_writing"] if bullish else chain["call_writing"]
            oi_confirmation = 1.0 if writing > 0 else 0.0
            strength = min(
                abs(float(minute_fast.iloc[-1] - minute_slow.iloc[-1]))
                / max(minute["atr"], 1.0),
                2.0,
            )
            momentum = min(abs(current_rsi - 50.0) / 20.0, 1.5)
            return {
                "index": name,
                "mode": "LONG",
                "option_type": direction,
                "score": 3.0 + strength + momentum + oi_confirmation,
                "spot": spot_price,
                "chain": chain,
                "candle": minute_candles[-1]["date"],
                "rsi": current_rsi,
                "adx": current_adx,
                "atr_percent": atr_percent,
            }

        return None

    def _find_contract(self, chain, option_type, strike):
        matches = chain[option_type]
        return min(matches, key=lambda row: abs(row["strike"] - strike), default=None)

    def _place_option_order(self, index_name, contract, transaction_type):
        config = self.indices[index_name]
        return self._place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=config["option_exchange"],
            tradingsymbol=contract["tradingsymbol"],
            transaction_type=transaction_type,
            quantity=self.execution_quantities[index_name],
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
            market_protection=-1,
        )

    def _place_order(self, **order_params):
        symbol = order_params["tradingsymbol"]
        if self._entry_in_progress and symbol not in self.active_trade_symbols:
            self.active_trade_pnl_baselines[symbol] = self._get_symbol_daily_pnl(
                symbol
            )
        order_id = self.kite.place_order(tag="ai_bot", **order_params)
        self.bot_traded_symbols.add(symbol)
        if self._entry_in_progress:
            self.active_trade_symbols.add(symbol)
        return order_id

    def _order_details(self, order_id):
        orders = self._kite_read("orders")
        return next(
            (order for order in orders if str(order.get("order_id")) == str(order_id)),
            None,
        )

    def _write_trade_audit(
        self,
        symbol,
        side,
        quantity,
        trigger_price,
        fill_price,
        exit_reason="",
        gross_pnl=None,
    ):
        slippage_points = (
            fill_price - trigger_price
            if side == self.kite.TRANSACTION_TYPE_BUY
            else trigger_price - fill_price
        )
        file_exists = os.path.exists(self.trade_audit_filename)
        with open(self.trade_audit_filename, "a", newline="") as audit_file:
            writer = csv.DictWriter(
                audit_file,
                fieldnames=[
                    "timestamp_ist",
                    "symbol",
                    "side",
                    "qty",
                    "trigger_signal_price",
                    "average_fill_price",
                    "slippage_points",
                    "slippage_rupees",
                    "exit_reason",
                    "gross_pnl",
                ],
            )
            if not file_exists or os.path.getsize(self.trade_audit_filename) == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_ist": datetime.now(
                        ZoneInfo("Asia/Kolkata")
                    ).isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": quantity,
                    "trigger_signal_price": f"{trigger_price:.2f}",
                    "average_fill_price": f"{fill_price:.2f}",
                    "slippage_points": f"{slippage_points:.2f}",
                    "slippage_rupees": f"{slippage_points * quantity:.2f}",
                    "exit_reason": exit_reason,
                    "gross_pnl": "" if gross_pnl is None else f"{gross_pnl:.2f}",
                }
            )

    def _audit_filled_order(
        self,
        order_id,
        symbol,
        side,
        requested_quantity,
        trigger_price,
        is_entry,
        exit_reason="",
        fallback_entry=None,
    ):
        order = self._order_details(order_id)
        if not order:
            raise RuntimeError(f"Order {order_id} was not found in Kite orders.")
        filled_quantity = int(order.get("filled_quantity") or 0)
        fill_price = float(order.get("average_price") or 0.0)
        if filled_quantity <= 0 or fill_price <= 0:
            return 0

        quantity = min(filled_quantity, requested_quantity)
        gross_pnl = None
        if is_entry:
            previous = self.open_trade_audit.get(symbol)
            if previous:
                combined_qty = previous["quantity"] + quantity
                previous["entry_price"] = (
                    previous["entry_price"] * previous["quantity"]
                    + fill_price * quantity
                ) / combined_qty
                previous["quantity"] = combined_qty
            else:
                self.open_trade_audit[symbol] = {
                    "side": side,
                    "entry_price": fill_price,
                    "quantity": quantity,
                }
        else:
            entry = self.open_trade_audit.get(symbol) or fallback_entry
            if entry:
                matched_quantity = min(quantity, entry["quantity"])
                price_change = (
                    fill_price - entry["entry_price"]
                    if entry["side"] == self.kite.TRANSACTION_TYPE_BUY
                    else entry["entry_price"] - fill_price
                )
                gross_pnl = price_change * matched_quantity
                remaining = entry["quantity"] - matched_quantity
                if remaining > 0 and symbol in self.open_trade_audit:
                    self.open_trade_audit[symbol]["quantity"] = remaining
                else:
                    self.open_trade_audit.pop(symbol, None)

        self._write_trade_audit(
            symbol,
            side,
            quantity,
            trigger_price,
            fill_price,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
        )
        return quantity

    def _submit_audited_entry(self, index_name, contract, side):
        quantity = self.execution_quantities[index_name]
        order_id = self._place_option_order(index_name, contract, side)
        if not self._wait_for_fill(order_id):
            try:
                self.kite.cancel_order(
                    variety=self.kite.VARIETY_REGULAR, order_id=order_id
                )
            except Exception:
                pass
        filled_quantity = self._audit_filled_order(
            order_id,
            contract["tradingsymbol"],
            side,
            quantity,
            float(contract["last_price"]),
            is_entry=True,
        )
        return order_id, filled_quantity

    def _wait_for_fill(self, order_id, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self.kite.order_history(order_id)
            status = history[-1].get("status") if history else None
            if status == "COMPLETE":
                return True
            if status in ("CANCELLED", "REJECTED"):
                return False
            time.sleep(1)
        return False

    def _execute_candidate(self, candidate):
        name = candidate["index"]
        chain = candidate["chain"]
        quantity = self.execution_quantities[name]

        if candidate["mode"] == "LONG":
            atm = round(candidate["spot"] / self.indices[name]["strike_step"]) * self.indices[name]["strike_step"]
            contract = self._find_contract(chain, candidate["option_type"], atm)
            if not contract or contract["last_price"] <= 0:
                return False
            order_id, filled_quantity = self._submit_audited_entry(
                name, contract, self.kite.TRANSACTION_TYPE_BUY
            )
            print(f"[ORDER] {name} BUY {contract['tradingsymbol']} x{quantity}; order {order_id}")
            return filled_quantity > 0

        if candidate["mode"] == "PUT_CREDIT":
            short_contract = chain["support"]
            if not short_contract:
                return False
            long_strike = short_contract["strike"] - self.indices[name]["strike_step"]
            option_type = "PE"
        else:
            short_contract = chain["resistance"]
            if not short_contract:
                return False
            long_strike = short_contract["strike"] + self.indices[name]["strike_step"]
            option_type = "CE"
        long_contract = self._find_contract(chain, option_type, long_strike)
        if (
            not short_contract
            or not long_contract
            or short_contract["last_price"] <= long_contract["last_price"]
        ):
            return False

        protective_order, protective_filled = self._submit_audited_entry(
            name, long_contract, self.kite.TRANSACTION_TYPE_BUY
        )
        if protective_filled < quantity:
            print(f"[SPREAD ABORTED] {name} protective leg was not confirmed filled.")
            if protective_filled:
                self._square_off_mis_positions("PER_TRADE_SL")
            return False

        try:
            daily_pnl = self._get_session_pnl()
        except Exception as exc:
            try:
                self._square_off_mis_positions("PER_TRADE_SL")
            except Exception as cleanup_exc:
                raise RuntimeError(
                    "Could not verify session P&L or close the protective spread leg."
                ) from cleanup_exc
            raise KiteReadError(
                "Could not verify session P&L; protective spread leg was closed."
            ) from exc
        if self._check_circuit_breaker(daily_pnl):
            return False

        try:
            short_order, short_filled = self._submit_audited_entry(
                name, short_contract, self.kite.TRANSACTION_TYPE_SELL
            )
        except Exception:
            self._square_off_mis_positions("PER_TRADE_SL")
            raise
        if short_filled < quantity:
            print(f"[SPREAD ABORTED] {name} short leg was not confirmed filled.")
            self._square_off_mis_positions("PER_TRADE_SL")
            return False
        print(
            f"[SPREAD] {name} {candidate['mode']} x{quantity}; "
            f"protective order {protective_order}, short order {short_order}"
        )
        return True

    @staticmethod
    def _position_pnl(position):
        pnl = position.get("pnl")
        if pnl is None:
            pnl = position.get("realised", 0.0) + position.get("unrealised", 0.0)
        return float(pnl or 0.0)

    def _get_symbol_daily_pnl(self, symbol):
        positions = self._kite_read("positions")
        return sum(
            self._position_pnl(position)
            for position in positions.get("day", [])
            if position.get("tradingsymbol") == symbol
        )

    def _get_daily_pnl(self):
        if not self.bot_traded_symbols:
            return 0.0

        positions = self._kite_read("positions")
        daily_pnl = 0.0
        for position in positions.get("day", []):
            if position.get("tradingsymbol") not in self.bot_traded_symbols:
                continue
            daily_pnl += self._position_pnl(position)
        return daily_pnl

    def _get_session_pnl(self):
        return self._get_daily_pnl()

    def _get_active_trade_pnl(self):
        if not self.active_trade_symbols:
            return 0.0
        positions = self._kite_read("positions")
        return sum(
            self._position_pnl(position)
            - self.active_trade_pnl_baselines.get(position.get("tradingsymbol"), 0.0)
            for position in positions.get("day", [])
            if position.get("tradingsymbol") in self.active_trade_symbols
        )

    def _get_live_bot_pnl(self):
        positions = self._kite_read("positions")
        open_positions = [
            position
            for position in positions.get("net", [])
            if position.get("product") == self.kite.PRODUCT_MIS
            and int(position.get("quantity") or 0) != 0
            and position.get("tradingsymbol") in self.bot_traded_symbols
        ]
        quote_keys = {
            f"{position['exchange']}:{position['tradingsymbol']}"
            for position in open_positions
        }
        quotes = self._kite_read("ltp", *quote_keys) if quote_keys else {}
        live_prices = {
            f"{position['exchange']}:{position['tradingsymbol']}": float(
                quotes[
                    f"{position['exchange']}:{position['tradingsymbol']}"
                ]["last_price"]
            )
            for position in open_positions
        }
        live_pnl_by_symbol = {}
        for position in open_positions:
            symbol = position["tradingsymbol"]
            quote_key = f"{position['exchange']}:{symbol}"
            live_pnl_by_symbol[symbol] = (
                live_prices[quote_key] - float(position.get("average_price") or 0.0)
            ) * int(position["quantity"])

        daily_pnl = 0.0
        trade_pnl = 0.0
        for position in positions.get("day", []):
            symbol = position.get("tradingsymbol")
            if symbol not in self.bot_traded_symbols:
                continue
            unrealised = float(position.get("unrealised") or 0.0)
            realised = position.get("realised")
            if realised is None:
                realised = self._position_pnl(position) - unrealised
            realised = float(realised or 0.0)
            live_unrealised = live_pnl_by_symbol.get(symbol, 0.0)
            daily_pnl += realised + live_unrealised
            if symbol in self.active_trade_symbols:
                trade_pnl += (
                    realised
                    - self.active_trade_pnl_baselines.get(symbol, 0.0)
                    + live_unrealised
                )
        return open_positions, live_prices, daily_pnl, trade_pnl

    def _get_open_bot_positions(self):
        if not self.bot_traded_symbols:
            return []
        positions = self._kite_read("positions")
        return [
            position
            for position in positions.get("net", [])
            if int(position.get("quantity") or 0) != 0
            and position.get("tradingsymbol") in self.bot_traded_symbols
        ]

    def _manage_active_trade(self, ist=None):
        ist = ist or ZoneInfo("Asia/Kolkata")
        monitored = False
        while True:
            now_ist = datetime.now(ist)
            market_close = now_ist.replace(hour=15, minute=15, second=0, microsecond=0)
            if now_ist >= market_close:
                try:
                    self._square_off_mis_positions("EOD_SQUARE_OFF")
                except Exception as exc:
                    print(f"⚠️ EOD square-off failed; retrying shortly: {exc}")
                    time.sleep(2)
                    continue
                return monitored

            try:
                open_positions, live_prices, daily_pnl, trade_pnl = (
                    self._get_live_bot_pnl()
                )
            except Exception as exc:
                print(f"⚠️ Active trade LTP/P&L check failed; retrying in 2 seconds: {exc}")
                time.sleep(2)
                continue
            if not open_positions:
                self.active_trade_symbols.clear()
                self.active_trade_pnl_baselines.clear()
                return monitored

            monitored = True
            print(
                f"[ACTIVE TRADE] P&L: ₹{trade_pnl:,.2f} | "
                f"Session P&L: ₹{daily_pnl:,.2f}"
            )
            if daily_pnl >= self.daily_target:
                reason = "DAILY_TARGET"
            elif daily_pnl <= -self.daily_max_loss:
                reason = "DAILY_SL"
            elif trade_pnl >= self.trade_target:
                reason = "PER_TRADE_TARGET"
            elif trade_pnl <= -self.trade_max_loss:
                reason = "PER_TRADE_SL"
            else:
                reason = None

            if reason:
                print(f"[ACTIVE TRADE] {reason} reached; closing bot positions.")
                try:
                    self._square_off_mis_positions(reason, live_prices)
                except Exception as exc:
                    print(f"⚠️ {reason} square-off failed; retrying shortly: {exc}")
                    time.sleep(2)
                else:
                    time.sleep(2)
                continue

            time.sleep(min(2, max(0, (market_close - datetime.now(ist)).total_seconds())))

    def _square_off_mis_positions(
        self, exit_reason="EOD_SQUARE_OFF", live_prices=None
    ):
        live_prices = live_prices or {}
        if not self.bot_traded_symbols:
            print("[SQUARE-OFF] No bot-traded symbols to close.")
            return

        positions = self._kite_read("positions")
        open_positions = [
            position
            for position in positions.get("net", [])
            if position.get("product") == self.kite.PRODUCT_MIS
            and int(position.get("quantity") or 0) != 0
            and position.get("tradingsymbol") in self.bot_traded_symbols
        ]
        if not open_positions:
            print("[SQUARE-OFF] No open MIS net positions.")
            return

        failures = []
        for position in open_positions:
            quantity = int(position["quantity"])
            transaction_type = (
                self.kite.TRANSACTION_TYPE_SELL
                if quantity > 0
                else self.kite.TRANSACTION_TYPE_BUY
            )
            try:
                order_id = self._place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=position["exchange"],
                    tradingsymbol=position["tradingsymbol"],
                    transaction_type=transaction_type,
                    quantity=abs(quantity),
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    market_protection=-1,
                )
                if not self._wait_for_fill(order_id):
                    try:
                        self.kite.cancel_order(
                            variety=self.kite.VARIETY_REGULAR, order_id=order_id
                        )
                    except Exception:
                        pass
                quote_key = f"{position['exchange']}:{position['tradingsymbol']}"
                trigger_price = live_prices.get(
                    quote_key, float(position.get("last_price") or 0.0)
                )
                filled_quantity = self._audit_filled_order(
                    order_id,
                    position["tradingsymbol"],
                    transaction_type,
                    abs(quantity),
                    trigger_price,
                    is_entry=False,
                    exit_reason=exit_reason,
                    fallback_entry={
                        "side": (
                            self.kite.TRANSACTION_TYPE_BUY
                            if quantity > 0
                            else self.kite.TRANSACTION_TYPE_SELL
                        ),
                        "entry_price": float(position.get("average_price") or 0.0),
                        "quantity": abs(quantity),
                    },
                )
                if filled_quantity == 0:
                    raise RuntimeError(
                        f"Square-off order {order_id} did not fill any quantity."
                    )
                print(
                    f"[SQUARE-OFF] {position['tradingsymbol']} "
                    f"{transaction_type} x{abs(quantity)}; order {order_id}"
                )
            except Exception as exc:
                failures.append(f"{position.get('tradingsymbol', 'unknown')}: {exc}")

        if failures:
            raise RuntimeError(
                "Could not submit all MIS square-off orders: " + "; ".join(failures)
            )

    def _square_off_if_due(self, ist):
        now_ist = datetime.now(ist)
        market_close = now_ist.replace(hour=15, minute=15, second=0, microsecond=0)
        if now_ist >= market_close:
            self._square_off_bot_positions("EOD_SQUARE_OFF")

    def _square_off_bot_positions(self, exit_reason="EOD_SQUARE_OFF"):
        for attempt in range(2):
            try:
                self._square_off_mis_positions(exit_reason)
                return
            except KiteReadError as exc:
                if attempt == 0:
                    print(f"⚠️ Square-off positions read failed; retrying shortly: {exc}")
                    time.sleep(1)
                else:
                    raise

    def _check_circuit_breaker(self, daily_pnl):
        if daily_pnl >= self.daily_target:
            print(f"\n🏆 [TARGET MET] Daily P&L ₹{daily_pnl:,.2f}. Stopping for the day.")
            return True
        if daily_pnl <= -self.daily_max_loss:
            print(f"\n🛑 [CIRCUIT BREAKER] Daily loss ₹{abs(daily_pnl):,.2f}. Stopping for the day.")
            return True
        return False

    def _daily_exit_reason(self, daily_pnl):
        if daily_pnl >= self.daily_target:
            return "DAILY_TARGET"
        if daily_pnl <= -self.daily_max_loss:
            return "DAILY_SL"
        return "EOD_SQUARE_OFF"
        
    def run_self_analysis(self):
        print("🧠 [AI Core] Codespace analysis running on historical database logs...")
        if not os.path.exists(self.log_filename):
            print("📝 No historical logs found yet. Booting standard risk parameters.")
            return
            
        try:
            with open(self.log_filename, "r") as f:
                logs = json.load(f)
            if len(logs) < 5:
                return
                
            recent_losses = len([t for t in logs[-10:] if t['pnl'] < 0])
            if recent_losses > 3:
                print("⚠️ System Warning: Recent loss cluster detected. Adjusting entry parameters.")
                self.rsi_overbought += 3.0
                self.rsi_oversold -= 3.0
                print(f"🔧 Triggers tightened dynamically to: {self.rsi_oversold} / {self.rsi_overbought}")
        except Exception as e:
            print(f"🔄 Memory load reset: {e}")

    def log_trade(self, strategy, pnl):
        trade_data = {"timestamp": time.time(), "strategy": strategy, "pnl": pnl}
        logs = []
        if os.path.exists(self.log_filename):
            try:
                with open(self.log_filename, "r") as f: logs = json.load(f)
            except: pass
        logs.append(trade_data)
        with open(self.log_filename, "w") as f:
            json.dump(logs, f, indent=4)

    def execute_live_cycle(self, cycles=None, poll_interval=30):
        try:
            self._run_live_cycle(cycles=cycles, poll_interval=poll_interval)
        finally:
            self._square_off_bot_positions("EOD_SQUARE_OFF")

    def _run_live_cycle(self, cycles=None, poll_interval=30):
        print("⚡ Live Kite session initiated. Scanning NIFTY, BANKNIFTY, and SENSEX...")
        ist = ZoneInfo("Asia/Kolkata")
        now_ist = datetime.now(ist)
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        if now_ist < market_open:
            print(f"[STANDBY] Waiting until {market_open:%H:%M} IST before scanning.")
            time.sleep((market_open - now_ist).total_seconds())

        now_ist = datetime.now(ist)
        market_close = now_ist.replace(hour=15, minute=15, second=0, microsecond=0)
        if now_ist >= market_close:
            print("[SESSION CLOSED] No new trades after 15:15 IST.")
            self._square_off_if_due(ist)
            return

        confirmation_counts = {}
        cycle = 0
        instruments_loaded = False

        while cycles is None or cycle < cycles:
            now_ist = datetime.now(ist)
            market_close = now_ist.replace(hour=15, minute=15, second=0, microsecond=0)
            if now_ist >= market_close:
                print("[SESSION CLOSED] No new trades after 15:15 IST.")
                break

            if not instruments_loaded:
                try:
                    self._load_market_instruments()
                    instruments_loaded = True
                except Exception as exc:
                    print(f"⚠️ Could not load market instruments; retrying next cycle: {exc}")
                    time.sleep(poll_interval)
                    continue

            try:
                daily_pnl = self._get_session_pnl()
            except Exception as exc:
                print(f"⚠️ Could not verify session P&L; retrying next cycle: {exc}")
                time.sleep(poll_interval)
                continue

            cycle += 1
            print(f"\n🔍 Cycle #{cycle} | Bot session P&L: ₹{daily_pnl:,.2f}")
            if self._check_circuit_breaker(daily_pnl):
                self._square_off_bot_positions(self._daily_exit_reason(daily_pnl))
                break
            try:
                if self._manage_active_trade(ist):
                    if cycles is None or cycle < cycles:
                        time.sleep(poll_interval)
                    continue
            except Exception as exc:
                print(f"⚠️ Could not manage the active bot trade; retrying next cycle: {exc}")
                if cycles is None or cycle < cycles:
                    time.sleep(poll_interval)
                continue

            candidates = []
            for name in self.indices:
                try:
                    candidate = self._analyze_index(name)
                    if (
                        candidate
                        and candidate["candle"] != self.last_order_candle.get(name)
                    ):
                        candidates.append(candidate)
                except Exception as exc:
                    print(f"⚠️ {name} scan failed: {exc}")

            setup_keys = {
                (candidate["index"], candidate["mode"], candidate.get("option_type"))
                for candidate in candidates
            }
            confirmation_counts = {
                key: confirmation_counts.get(key, 0) + 1 for key in setup_keys
            }
            candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
            for candidate in candidates:
                setup_key = (
                    candidate["index"],
                    candidate["mode"],
                    candidate.get("option_type"),
                )
                print(
                    f"[SETUP] {candidate['index']} {candidate['mode']} "
                    f"score={candidate['score']:.2f} "
                    f"confirmations={confirmation_counts[setup_key]}/3 "
                    f"RSI={candidate['rsi']:.1f} ADX={candidate['adx']:.1f} "
                    f"ATR%={candidate['atr_percent']:.3%}"
                )

            scan_time = datetime.now(ist)
            warmup_end = scan_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = scan_time.replace(hour=15, minute=15, second=0, microsecond=0)
            if scan_time >= market_close:
                print("[SESSION CLOSED] No new trades after 15:15 IST.")
                break

            if scan_time < warmup_end:
                print("[WARMUP] Scanning only; order placement is disabled until 09:30 IST.")
                eligible_candidates = []
            else:
                eligible_candidates = [
                    candidate
                    for candidate in candidates
                    if confirmation_counts[
                        (
                            candidate["index"],
                            candidate["mode"],
                            candidate.get("option_type"),
                        )
                    ]
                    >= 3
                ]
            if not eligible_candidates and scan_time >= warmup_end:
                if candidates:
                    print("Waiting for three consecutive confirmations.")
                else:
                    print("No qualified multi-index setup this cycle.")

            for candidate in eligible_candidates:
                order_time = datetime.now(ist)
                order_cutoff = order_time.replace(
                    hour=15, minute=15, second=0, microsecond=0
                )
                order_warmup_end = order_time.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                if order_time >= order_cutoff:
                    print("[SESSION CLOSED] No new trades after 15:15 IST.")
                    break
                if order_time < order_warmup_end:
                    print("[WARMUP] Scanning only; order placement is disabled until 09:30 IST.")
                    break

                try:
                    open_positions = self._get_open_bot_positions()
                except Exception as exc:
                    print(f"⚠️ Could not verify open bot positions; retrying next cycle: {exc}")
                    break
                if open_positions:
                    print("[TRADE SKIPPED] An existing bot MIS position is still open.")
                    break

                try:
                    # Check bot-session P&L again immediately before trading.
                    daily_pnl = self._get_session_pnl()
                except Exception as exc:
                    print(f"⚠️ Could not verify session P&L; retrying next cycle: {exc}")
                    self._square_off_if_due(ist)
                    break
                if self._check_circuit_breaker(daily_pnl):
                    self._square_off_bot_positions(
                        self._daily_exit_reason(daily_pnl)
                    )
                    return
                order_time = datetime.now(ist)
                order_cutoff = order_time.replace(
                    hour=15, minute=15, second=0, microsecond=0
                )
                if order_time >= order_cutoff:
                    print("[SESSION CLOSED] No new trades after 15:15 IST.")
                    break

                name = candidate["index"]
                try:
                    self._entry_in_progress = True
                    try:
                        submitted = self._execute_candidate(candidate)
                    finally:
                        self._entry_in_progress = False
                except KiteReadError as exc:
                    print(f"⚠️ {name} scan paused; retrying next cycle: {exc}")
                    try:
                        self._manage_active_trade(ist)
                    except Exception as monitor_exc:
                        print(
                            "⚠️ Could not manage a position after the order error: "
                            f"{monitor_exc}"
                        )
                    break
                except Exception as exc:
                    print(f"⚠️ {name} order failed; continuing scan: {exc}")
                    try:
                        self._manage_active_trade(ist)
                    except Exception as monitor_exc:
                        print(
                            "⚠️ Could not manage a position after the order error: "
                            f"{monitor_exc}"
                        )
                    continue
                if submitted:
                    self.last_order_candle[name] = candidate["candle"]
                try:
                    if self._manage_active_trade(ist):
                        break
                except Exception as exc:
                    print(f"⚠️ Could not verify active trade P&L; retrying next cycle: {exc}")
                    break
                try:
                    daily_pnl = self._get_session_pnl()
                except Exception as exc:
                    print(f"⚠️ Could not verify session P&L; retrying next cycle: {exc}")
                    self._square_off_if_due(ist)
                    break
                if self._check_circuit_breaker(daily_pnl):
                    self._square_off_bot_positions(
                        self._daily_exit_reason(daily_pnl)
                    )
                    return
                if submitted:
                    break

            cycle_end = datetime.now(ist)
            if cycle_end >= cycle_end.replace(
                hour=15, minute=15, second=0, microsecond=0
            ):
                print("[SESSION CLOSED] No new trades after 15:15 IST.")
                break

            if cycles is not None and cycle >= cycles:
                break

            time.sleep(poll_interval)

        self._square_off_if_due(ist)

if __name__ == "__main__":
    bot = CodespaceSelfImprovingRobot()
    bot.execute_live_cycle()
