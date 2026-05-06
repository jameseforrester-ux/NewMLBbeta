"""
Polymarket integration — scans MLB markets, reads odds, calculates implied probs.
Uses the Gamma API (market discovery) and CLOB API (order book).
"""

import aiohttp
import asyncio
import re
import logging
from datetime import datetime
from typing import Optional

from config import POLY_GAMMA_API, POLY_CLOB_API

log = logging.getLogger("polymarket")

# ── Cache ────────────────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 120  # 2 minutes for market data


def _cache_get(key: str):
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.utcnow() - ts).total_seconds() < CACHE_TTL:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val):
    _cache[key] = (val, datetime.utcnow())


async def _fetch_json(url: str, params: dict = None) -> dict | list:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
                log.warning(f"HTTP {resp.status} from {url}")
                return {}
    except Exception as e:
        log.error(f"Polymarket fetch error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET DISCOVERY (Gamma API)
# ═══════════════════════════════════════════════════════════════════════════

async def search_mlb_markets(query: str = "MLB") -> list[dict]:
    """Search for MLB-related markets on Polymarket."""
    cache_key = f"poly_search_{query}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = await _fetch_json(
        f"{POLY_GAMMA_API}/markets",
        {
            "tag": "sports",
            "closed": "false",
            "limit": 100,
        }
    )

    if not isinstance(data, list):
        data = data.get("data", []) if isinstance(data, dict) else []

    mlb_markets = []
    mlb_keywords = [
        "mlb", "baseball", "yankees", "dodgers", "braves", "astros",
        "mets", "phillies", "cubs", "red sox", "cardinals", "giants",
        "padres", "orioles", "guardians", "rangers", "mariners",
        "twins", "rays", "brewers", "diamondbacks", "reds", "royals",
        "tigers", "angels", "pirates", "rockies", "marlins",
        "white sox", "nationals", "athletics", "blue jays",
        "world series", "home run", "strikeout", "no-hitter",
        "runs", "over under", "moneyline", "run line",
    ]

    for market in data:
        question = (market.get("question", "") or "").lower()
        desc = (market.get("description", "") or "").lower()
        tags = [t.lower() for t in (market.get("tags", []) or [])]
        combined = f"{question} {desc} {' '.join(tags)}"

        if any(kw in combined for kw in mlb_keywords) or "mlb" in tags:
            parsed = _parse_market(market)
            if parsed:
                mlb_markets.append(parsed)

    _cache_set(cache_key, mlb_markets)
    return mlb_markets


async def get_market_by_id(market_id: str) -> Optional[dict]:
    """Get a specific market by its condition ID or slug."""
    data = await _fetch_json(f"{POLY_GAMMA_API}/markets/{market_id}")
    if data:
        return _parse_market(data)
    return None


async def get_todays_mlb_markets() -> list[dict]:
    """Get all MLB markets for today's games."""
    all_markets = await search_mlb_markets("MLB")

    today = datetime.now().strftime("%Y-%m-%d")
    todays = []
    for m in all_markets:
        # Filter for today's games by checking end date or question text
        end_date = m.get("end_date", "")
        question = m.get("question", "").lower()

        # Check for today's date references or active markets
        if today in end_date or m.get("active", False):
            todays.append(m)

    return todays if todays else all_markets[:20]


def _parse_market(market: dict) -> Optional[dict]:
    """Parse a raw Gamma API market into a clean dict."""
    if not market:
        return None

    outcomes = market.get("outcomes", []) or []
    tokens = market.get("tokens", []) or market.get("clobTokenIds", []) or []

    # Parse outcome prices
    outcome_data = []
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, str):
            price = None
            if "outcomePrices" in market:
                prices = market.get("outcomePrices", [])
                if isinstance(prices, str):
                    try:
                        import json
                        prices = json.loads(prices)
                    except:
                        prices = []
                if i < len(prices):
                    try:
                        price = float(prices[i])
                    except (ValueError, TypeError):
                        pass
            outcome_data.append({
                "name":         outcome,
                "price":        price,
                "token_id":     tokens[i] if i < len(tokens) else None,
                "implied_prob": price if price else None,
            })
        elif isinstance(outcome, dict):
            price = outcome.get("price")
            if price:
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    price = None
            outcome_data.append({
                "name":         outcome.get("value", outcome.get("name", f"Outcome {i}")),
                "price":        price,
                "token_id":     outcome.get("clobTokenId", tokens[i] if i < len(tokens) else None),
                "implied_prob": price,
            })

    market_type = _classify_market(market.get("question", ""))

    return {
        "id":           market.get("id", market.get("conditionId", "")),
        "question":     market.get("question", ""),
        "description":  market.get("description", ""),
        "slug":         market.get("slug", ""),
        "active":       market.get("active", True),
        "closed":       market.get("closed", False),
        "volume":       market.get("volume", 0),
        "liquidity":    market.get("liquidity", 0),
        "end_date":     market.get("endDate", market.get("end_date_iso", "")),
        "outcomes":     outcome_data,
        "market_type":  market_type,
        "url":          f"https://polymarket.com/event/{market.get('slug', '')}",
    }


def _classify_market(question: str) -> str:
    """Classify market type from question text."""
    q = question.lower()
    if any(kw in q for kw in ["over", "under", "total runs", "total score"]):
        return "over_under"
    elif any(kw in q for kw in ["run line", "spread", "handicap"]):
        return "run_line"
    elif any(kw in q for kw in ["no run", "first inning", "scoreless", "nrfi", "yrfi"]):
        return "first_inning"
    elif any(kw in q for kw in ["win", "beat", "defeat", "winner", "vs", "v."]):
        return "moneyline"
    elif any(kw in q for kw in ["home run", "strikeout", "hit", "rbi"]):
        return "prop"
    else:
        return "other"


# ═══════════════════════════════════════════════════════════════════════════
#  ORDER BOOK (CLOB API)
# ═══════════════════════════════════════════════════════════════════════════

async def get_order_book(token_id: str) -> dict:
    """Get order book for a specific outcome token."""
    data = await _fetch_json(
        f"{POLY_CLOB_API}/book",
        {"token_id": token_id}
    )
    return data


async def get_market_price(token_id: str) -> Optional[float]:
    """Get the current mid price for a token."""
    book = await get_order_book(token_id)
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if bids and asks:
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0))
        return (best_bid + best_ask) / 2
    elif bids:
        return float(bids[0].get("price", 0))
    elif asks:
        return float(asks[0].get("price", 0))
    return None


async def get_market_implied_odds(market: dict) -> dict:
    """Calculate implied probabilities from market prices."""
    result = {"market_id": market.get("id"), "outcomes": []}

    for outcome in market.get("outcomes", []):
        price = outcome.get("price")
        if price is None and outcome.get("token_id"):
            price = await get_market_price(outcome["token_id"])

        implied_prob = price if price else None
        result["outcomes"].append({
            "name":         outcome.get("name"),
            "price":        price,
            "implied_prob": implied_prob,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  LINE MOVEMENT TRACKING
# ═══════════════════════════════════════════════════════════════════════════

_line_history: dict[str, list[tuple[datetime, float]]] = {}


async def track_line_movement(market_id: str, outcomes: list[dict]):
    """Record current prices for line movement detection."""
    now = datetime.utcnow()
    for outcome in outcomes:
        key = f"{market_id}_{outcome.get('name', '')}"
        if key not in _line_history:
            _line_history[key] = []
        price = outcome.get("price")
        if price is not None:
            _line_history[key].append((now, price))
            # Keep last 100 entries
            _line_history[key] = _line_history[key][-100:]


def get_line_movement(market_id: str, outcome_name: str) -> dict:
    """Get line movement data for a specific outcome."""
    key = f"{market_id}_{outcome_name}"
    history = _line_history.get(key, [])

    if len(history) < 2:
        return {"movement": 0, "direction": "stable", "data_points": len(history)}

    current = history[-1][1]
    previous = history[-2][1]
    first = history[0][1]

    return {
        "current":      current,
        "previous":     previous,
        "open":         first,
        "change_last":  round((current - previous) * 100, 2),
        "change_open":  round((current - first) * 100, 2),
        "direction":    "up" if current > previous else "down" if current < previous else "stable",
        "data_points":  len(history),
    }
