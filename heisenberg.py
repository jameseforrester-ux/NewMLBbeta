"""
Heisenberg Prediction Market Intelligence — direct REST API wrapper.
Calls narrative.agent.heisenberg.so from the VPS bot process.

Agents used:
  574 — Polymarket Markets    (find MLB market condition_ids + token_ids)
  575 — Market 360            (whale concentration, volume trend, winning side)
  556 — Polymarket Trades     (individual trade-level data by wallet)
  584 — Heisenberg Leaderboard(elite wallet list with H-Score)
  596 — Price Jump Detection  (sharp money entering a market)
  568 — Candlesticks          (price history for momentum)
  585 — Social Pulse          (Twitter/social sentiment)
"""

import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("heisenberg")

HEISENBERG_API = "https://narrative.agent.heisenberg.so/api/v2/semantic/retrieve/parameterized"

# ── Agent IDs ────────────────────────────────────────────────────────────────
AGENT_MARKETS      = 574
AGENT_MARKET_360   = 575
AGENT_TRADES       = 556
AGENT_H_LEADERBOARD = 584
AGENT_PRICE_JUMPS  = 596
AGENT_CANDLESTICKS = 568
AGENT_SOCIAL_PULSE = 585

# ── Cache ────────────────────────────────────────────────────────────────────
_cache: dict = {}
_ELITE_WALLETS_CACHE: Optional[list] = None
_ELITE_WALLETS_TS: Optional[datetime] = None


def _cache_get(key: str, ttl: int = 300):
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.utcnow() - ts).total_seconds() < ttl:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val):
    _cache[key] = (val, datetime.utcnow())


# ═══════════════════════════════════════════════════════════════════════════
#  CORE HTTP HELPER
# ═══════════════════════════════════════════════════════════════════════════

async def _query(agent_id: int, params: dict,
                 limit: int = 20, offset: int = 0) -> list:
    """
    POST to Heisenberg parameterized retrieval endpoint.
    Returns the results list or empty list on error.
    """
    from config import HEISENBERG_TOKEN
    headers = {
        "Authorization": f"Bearer {HEISENBERG_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "agent_id": agent_id,
        "params": params,
        "pagination": {"limit": limit, "offset": offset},
        "formatter_config": {"format_type": "raw"},
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                HEISENBERG_API,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"Heisenberg HTTP {resp.status} agent={agent_id}")
                    return []
                data = await resp.json()
                # Handle both direct list and nested data.results
                if isinstance(data, list):
                    return data
                return data.get("data", {}).get("results", [])
    except Exception as e:
        log.error(f"Heisenberg query error agent={agent_id}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 574 — FIND MLB MARKETS
# ═══════════════════════════════════════════════════════════════════════════

async def find_mlb_markets(keyword: str = "mlb",
                            min_volume: float = 0,
                            closed: bool = False) -> list[dict]:
    """Find Polymarket markets matching an MLB keyword."""
    cache_key = f"markets_{keyword}_{min_volume}_{closed}"
    cached = _cache_get(cache_key, ttl=180)
    if cached is not None:
        return cached

    results = await _query(AGENT_MARKETS, {
        "market_slug": keyword,
        "min_volume": str(min_volume),
        "closed": str(closed),
    }, limit=50)

    _cache_set(cache_key, results)
    return results


async def find_game_markets(away_team: str, home_team: str) -> list[dict]:
    """
    Find all Polymarket markets for a specific game.
    Tries team name keywords and MLB event slug patterns.
    """
    # Build search keywords from last word of team name (e.g. "Yankees", "Dodgers")
    away_kw = away_team.lower().split()[-1]
    home_kw = home_team.lower().split()[-1]

    results = []
    for kw in [away_kw, home_kw, "mlb"]:
        markets = await find_mlb_markets(keyword=kw)
        for m in markets:
            q = (m.get("question") or "").lower()
            slug = (m.get("slug") or "").lower()
            combined = q + " " + slug
            # Must mention both teams to be specific to this game
            if away_kw in combined or home_kw in combined:
                if m not in results:
                    results.append(m)
        await asyncio.sleep(0.1)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 575 — MARKET 360 (structure + winning side)
# ═══════════════════════════════════════════════════════════════════════════

async def get_market_360(condition_id: str) -> dict:
    """
    Full structural analysis of a market.
    Returns whale concentration, volume trend, and crucially:
    which side (Yes/No) is currently profitable.
    """
    cache_key = f"m360_{condition_id}"
    cached = _cache_get(cache_key, ttl=120)
    if cached is not None:
        return cached

    results = await _query(AGENT_MARKET_360, {
        "condition_id": condition_id,
        "volume_trend": "ALL",
        "min_volume_24h": "0",
        "min_liquidity_percentile": "0",
        "min_top1_wallet_pct": "0",
        "max_unique_traders_7d": "0",
    }, limit=1)

    result = results[0] if results else {}
    _cache_set(cache_key, result)
    return result


def interpret_market_360(m360: dict) -> dict:
    """
    Derive a betting signal from Market 360 data.

    Returns:
      signal:       'yes' | 'no' | 'neutral'
      confidence:   0.0 - 1.0
      reasoning:    list of strings explaining the signal
      risk_flags:   list of active risk flags
    """
    if not m360:
        return {"signal": "neutral", "confidence": 0.0,
                "reasoning": ["No market data"], "risk_flags": []}

    signal = "neutral"
    confidence = 0.0
    reasoning = []
    risk_flags = []

    # ── Winning side ──────────────────────────────────────────────────────
    yes_pnl = float(m360.get("yes_avg_pnl") or 0)
    no_pnl  = float(m360.get("no_avg_pnl") or 0)
    winning_side = m360.get("winning_side")

    if winning_side == "yes" and yes_pnl > 0:
        signal = "yes"
        confidence += 0.30
        reasoning.append(f"YES side is profitable (avg PnL ${yes_pnl:.2f})")
    elif winning_side == "no" and no_pnl > 0:
        signal = "no"
        confidence += 0.30
        reasoning.append(f"NO side is profitable (avg PnL ${no_pnl:.2f})")
    elif yes_pnl > 0 or no_pnl > 0:
        pnl_ratio = float(m360.get("profit_loss_ratio") or 0)
        if pnl_ratio > 1.5:
            reasoning.append(f"Profit/loss ratio: {pnl_ratio:.1f}x")
            confidence += 0.10

    # ── Whale concentration ───────────────────────────────────────────────
    top1_pct  = float(m360.get("top1_wallet_pct") or 0)
    top3_pct  = float(m360.get("top3_wallet_pct") or 0)
    top10_pct = float(m360.get("top10_wallet_pct") or 0)

    whale_flag = m360.get("whale_control_flag", False)
    if whale_flag or top1_pct > 40:
        risk_flags.append(f"Whale controlled: top wallet holds {top1_pct:.0f}%")
        confidence -= 0.10  # whale = manipulation risk
    elif top10_pct < 30:
        reasoning.append("Distributed liquidity — organic market")
        confidence += 0.10

    # ── Volume trend ──────────────────────────────────────────────────────
    vol_trend = m360.get("volume_trend", "")
    if vol_trend == "Spiking":
        reasoning.append("Volume spiking — sharp money entering")
        confidence += 0.20
    elif vol_trend == "No Trades":
        reasoning.append("No recent trading activity — thin market")
        confidence -= 0.15
    elif vol_trend == "Dying Interest":
        risk_flags.append("Volume dying — market losing interest")
        confidence -= 0.10

    # ── Risk flags ────────────────────────────────────────────────────────
    if m360.get("liquidity_risk_flag"):
        risk_flags.append("Low liquidity — slippage risk")
    if m360.get("squeeze_risk_flag"):
        risk_flags.append("Squeeze risk detected")
    if m360.get("volume_collapse_risk_flag"):
        risk_flags.append("Volume collapse risk")

    return {
        "signal":     signal,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning":  reasoning,
        "risk_flags": risk_flags,
        "raw": {
            "yes_avg_pnl":    yes_pnl,
            "no_avg_pnl":     no_pnl,
            "top1_wallet_pct": top1_pct,
            "volume_trend":   vol_trend,
            "winning_side":   winning_side,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 584 — ELITE WALLET LIST (cached 4 hours)
# ═══════════════════════════════════════════════════════════════════════════

async def get_elite_wallets(min_win_rate: float = 0.55,
                             min_roi: float = 0.12,
                             limit: int = 50) -> list[str]:
    """
    Fetch Elite/Sharp tier wallet addresses from Heisenberg leaderboard.
    Cached for 4 hours — doesn't change frequently.
    """
    global _ELITE_WALLETS_CACHE, _ELITE_WALLETS_TS

    if (_ELITE_WALLETS_CACHE is not None and _ELITE_WALLETS_TS is not None
            and (datetime.utcnow() - _ELITE_WALLETS_TS).total_seconds() < 14400):
        return _ELITE_WALLETS_CACHE

    results = await _query(AGENT_H_LEADERBOARD, {
        "min_win_rate_15d": str(min_win_rate),
        "min_roi_15d":      str(min_roi),
        "min_total_trades_15d": "20",
        "max_win_rate_15d": "0.95",  # filter bots
        "sort_by": "h_score",
    }, limit=limit)

    wallets = [r["wallet"] for r in results if r.get("wallet")]
    _ELITE_WALLETS_CACHE = wallets
    _ELITE_WALLETS_TS = datetime.utcnow()
    log.info(f"Loaded {len(wallets)} elite wallets from Heisenberg")
    return wallets


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 556 — TRADES (smart money tracker)
# ═══════════════════════════════════════════════════════════════════════════

async def get_smart_money_signal(condition_id: str,
                                  hours_back: int = 24) -> dict:
    """
    Check if elite wallets have been trading this market.
    Returns which side they're on and their aggregate conviction.

    This is the most powerful signal in the model — when smart money
    aligns with your statistical edge, confidence jumps significantly.
    """
    cache_key = f"smart_{condition_id}_{hours_back}"
    cached = _cache_get(cache_key, ttl=300)
    if cached is not None:
        return cached

    elite_wallets = await get_elite_wallets()
    if not elite_wallets:
        return {"signal": "neutral", "confidence": 0.0,
                "elite_trades": [], "available": False}

    now_ts = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())

    # Pull all recent trades on this market
    trades = await _query(AGENT_TRADES, {
        "condition_id": condition_id,
        "proxy_wallet":  "ALL",
        "start_time":    str(start_ts),
        "end_time":      str(now_ts),
    }, limit=100)

    if not trades:
        result = {"signal": "neutral", "confidence": 0.0,
                  "elite_trades": [], "available": True,
                  "total_trades": 0}
        _cache_set(cache_key, result)
        return result

    # Filter for elite wallet trades
    elite_set = set(w.lower() for w in elite_wallets)
    elite_trades = [
        t for t in trades
        if (t.get("proxy_wallet") or "").lower() in elite_set
    ]

    if not elite_trades:
        result = {"signal": "neutral", "confidence": 0.0,
                  "elite_trades": [], "available": True,
                  "total_trades": len(trades)}
        _cache_set(cache_key, result)
        return result

    # Tally which side elite money is on
    yes_volume = sum(
        float(t.get("size") or 0) * float(t.get("price") or 0)
        for t in elite_trades
        if (t.get("outcome") or "").lower() in ("yes", "yes run", "over", "a")
    )
    no_volume = sum(
        float(t.get("size") or 0) * float(t.get("price") or 0)
        for t in elite_trades
        if (t.get("outcome") or "").lower() in ("no", "no run", "under", "b")
    )
    total_elite_vol = yes_volume + no_volume

    signal = "neutral"
    confidence = 0.0
    dominant_side = None

    if total_elite_vol > 0:
        yes_share = yes_volume / total_elite_vol
        if yes_share >= 0.65:
            signal = "yes"
            dominant_side = "YES"
            confidence = min(0.45, 0.20 + (yes_share - 0.65) * 0.8)
        elif yes_share <= 0.35:
            signal = "no"
            dominant_side = "NO"
            confidence = min(0.45, 0.20 + (0.35 - yes_share) * 0.8)

    result = {
        "signal":        signal,
        "confidence":    round(confidence, 3),
        "dominant_side": dominant_side,
        "elite_trades":  len(elite_trades),
        "total_trades":  len(trades),
        "yes_volume":    round(yes_volume, 2),
        "no_volume":     round(no_volume, 2),
        "yes_share":     round(yes_volume / total_elite_vol, 3) if total_elite_vol > 0 else 0.5,
        "available":     True,
    }
    _cache_set(cache_key, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 596 — PRICE JUMP DETECTION (sharp money signal)
# ═══════════════════════════════════════════════════════════════════════════

async def detect_price_jumps(token_id: str,
                              hours_back: int = 6,
                              min_change_pct: float = 5.0) -> list[dict]:
    """
    Detect sharp price movements on a specific outcome token.
    A sudden 5%+ move in a thin MLB market = someone knows something.
    """
    cache_key = f"jumps_{token_id}_{hours_back}"
    cached = _cache_get(cache_key, ttl=120)
    if cached is not None:
        return cached

    results = await _query(AGENT_PRICE_JUMPS, {
        "token_id":       token_id,
        "resolution":     "15m",
        "min_change_pct": str(min_change_pct),
        "lookback_hours": str(hours_back),
    }, limit=20)

    _cache_set(cache_key, results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 568 — CANDLESTICKS (price momentum)
# ═══════════════════════════════════════════════════════════════════════════

async def get_price_momentum(token_id: str, hours_back: int = 12) -> dict:
    """
    Get recent price trend for an outcome token.
    Returns direction and magnitude of momentum.
    """
    cache_key = f"momentum_{token_id}_{hours_back}"
    cached = _cache_get(cache_key, ttl=300)
    if cached is not None:
        return cached

    now_ts = int(datetime.utcnow().timestamp())
    start_ts = now_ts - (hours_back * 3600)

    candles = await _query(AGENT_CANDLESTICKS, {
        "token_id":   token_id,
        "interval":   "1h",
        "start_time": str(start_ts),
        "end_time":   str(now_ts),
    }, limit=24)

    if not candles:
        return {"direction": "neutral", "change_pct": 0.0, "available": False}

    prices = [float(c.get("close") or 0) for c in candles if c.get("close")]
    if len(prices) < 2:
        return {"direction": "neutral", "change_pct": 0.0, "available": False}

    open_price  = prices[0]
    close_price = prices[-1]
    change_pct  = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0

    result = {
        "direction":   "up" if change_pct > 1 else "down" if change_pct < -1 else "flat",
        "change_pct":  round(change_pct, 2),
        "open_price":  round(open_price, 4),
        "close_price": round(close_price, 4),
        "candles":     len(prices),
        "available":   True,
    }
    _cache_set(cache_key, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 585 — SOCIAL PULSE
# ═══════════════════════════════════════════════════════════════════════════

async def get_social_sentiment(away_team: str, home_team: str,
                                hours_back: int = 12) -> dict:
    """
    Get social media sentiment for teams playing today.
    Price-sentiment divergence = exploitable edge.
    """
    away_kw = away_team.split()[-1]
    home_kw = home_team.split()[-1]
    keywords = "{" + f"{away_kw},{home_kw},MLB,baseball" + "}"

    cache_key = f"social_{away_kw}_{home_kw}_{hours_back}"
    cached = _cache_get(cache_key, ttl=600)
    if cached is not None:
        return cached

    results = await _query(AGENT_SOCIAL_PULSE, {
        "keywords":   keywords,
        "hours_back": str(hours_back),
    }, limit=20)

    if not results:
        return {"available": False, "sentiment": "neutral", "volume": 0}

    # Aggregate sentiment across posts
    total_likes = sum(int(r.get("like_count") or 0) for r in results)
    total_rt    = sum(int(r.get("retweet_count") or 0) for r in results)
    accel_vals  = [float(r.get("acceleration") or 0) for r in results if r.get("acceleration")]
    avg_accel   = sum(accel_vals) / len(accel_vals) if accel_vals else 0

    result = {
        "available":    True,
        "post_count":   len(results),
        "total_likes":  total_likes,
        "total_rt":     total_rt,
        "acceleration": round(avg_accel, 2),
        "sentiment":    "bullish" if avg_accel > 1.5 else "bearish" if avg_accel < -1.5 else "neutral",
        "top_posts":    results[:3],
    }
    _cache_set(cache_key, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSITE HEISENBERG SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

async def get_full_market_signal(market: dict) -> dict:
    """
    Run all Heisenberg intelligence on one market.
    Returns a combined signal with confidence score.

    Combines:
      - Market 360 (winning side, whale control, volume trend)
      - Smart money (elite wallet positioning)
      - Price jumps (sudden sharp moves)
      - Price momentum
    """
    condition_id   = market.get("condition_id", "")
    side_a_token   = market.get("side_a_token_id", "")
    side_b_token   = market.get("side_b_token_id", "")
    side_a_outcome = (market.get("side_a_outcome") or "Yes").lower()
    side_b_outcome = (market.get("side_b_outcome") or "No").lower()

    if not condition_id:
        return {"available": False}

    # Run analyses concurrently
    m360_task    = get_market_360(condition_id)
    smart_task   = get_smart_money_signal(condition_id)
    jumps_a_task = detect_price_jumps(side_a_token) if side_a_token else asyncio.sleep(0)
    jumps_b_task = detect_price_jumps(side_b_token) if side_b_token else asyncio.sleep(0)
    mom_a_task   = get_price_momentum(side_a_token) if side_a_token else asyncio.sleep(0)

    m360, smart, jumps_a, jumps_b, momentum = await asyncio.gather(
        m360_task, smart_task, jumps_a_task, jumps_b_task, mom_a_task,
        return_exceptions=True
    )

    # Safely handle exceptions
    m360     = m360     if not isinstance(m360, Exception)     else {}
    smart    = smart    if not isinstance(smart, Exception)    else {}
    jumps_a  = jumps_a  if not isinstance(jumps_a, Exception)  else []
    jumps_b  = jumps_b  if not isinstance(jumps_b, Exception)  else []
    momentum = momentum if not isinstance(momentum, Exception) else {}

    m360_signal  = interpret_market_360(m360)

    # ── Combine signals ───────────────────────────────────────────────────
    signals = []

    # Market 360 signal (weighted 0.30)
    if m360_signal["signal"] != "neutral":
        signals.append((m360_signal["signal"], m360_signal["confidence"] * 0.30))

    # Smart money signal (weighted 0.45 — most important)
    if isinstance(smart, dict) and smart.get("signal") != "neutral":
        signals.append((smart["signal"], smart.get("confidence", 0) * 0.45))

    # Price jump signal (weighted 0.25)
    recent_jumps_a = [j for j in (jumps_a or []) if isinstance(j, dict) and j.get("direction") == "up"]
    recent_jumps_b = [j for j in (jumps_b or []) if isinstance(j, dict) and j.get("direction") == "up"]

    if recent_jumps_a:
        biggest = max(abs(float(j.get("change_pct", 0))) for j in recent_jumps_a)
        jump_conf = min(0.25, biggest / 100)
        signals.append(("yes", jump_conf * 0.25))

    if recent_jumps_b:
        biggest = max(abs(float(j.get("change_pct", 0))) for j in recent_jumps_b)
        jump_conf = min(0.25, biggest / 100)
        signals.append(("no", jump_conf * 0.25))

    # Tally final signal
    yes_score = sum(conf for sig, conf in signals if sig == "yes")
    no_score  = sum(conf for sig, conf in signals if sig == "no")

    if yes_score > no_score and yes_score > 0.05:
        final_signal = side_a_outcome  # map "yes" → actual outcome name
        final_conf   = yes_score
    elif no_score > yes_score and no_score > 0.05:
        final_signal = side_b_outcome
        final_conf   = no_score
    else:
        final_signal = "neutral"
        final_conf   = 0.0

    return {
        "available":      True,
        "signal":         final_signal,
        "confidence":     round(final_conf, 3),
        "market_360":     m360_signal,
        "smart_money":    smart if isinstance(smart, dict) else {},
        "price_jumps_a":  len(recent_jumps_a),
        "price_jumps_b":  len(recent_jumps_b),
        "momentum":       momentum if isinstance(momentum, dict) else {},
        "risk_flags":     m360_signal.get("risk_flags", []),
        "reasoning":      m360_signal.get("reasoning", []),
    }
