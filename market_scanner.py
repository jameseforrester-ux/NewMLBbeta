"""
market_scanner.py — Full-sweep Polymarket MLB market scanner.

For every game on today's slate, finds and evaluates ALL 4 market types:
  1. Moneyline      (who wins)
  2. Over/Under     (total runs)
  3. Run Line       (spread, usually ±1.5)
  4. NRFI/YRFI      (no/yes run first inning)

Pulls markets via Heisenberg, matches to games, runs the 4-layer model
against each outcome, and returns a unified edge report.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from config import E

log = logging.getLogger("scanner")

# ── Market type keyword matchers ─────────────────────────────────────────────
MARKET_PATTERNS = {
    "moneyline": [
        r"\bwin\b", r"\bwinner\b", r"\bbeat\b", r"\bdefeats?\b",
        r"\bvs\.?\b", r"\bmoneyline\b", r"\bml\b",
    ],
    "over_under": [
        r"\bover\b", r"\bunder\b", r"\btotal runs?\b", r"\bo/u\b",
        r"\bmore than\b", r"\bfewer than\b", r"\btotal score\b",
    ],
    "run_line": [
        r"\brun line\b", r"\bspread\b", r"\b[+-]1\.5\b", r"\bhandicap\b",
        r"\bcover\b", r"\bby \d\b",
    ],
    "nrfi": [
        r"\bnrfi\b", r"\byrfi\b", r"\bno run.{0,15}first\b",
        r"\bfirst inning\b", r"\bscoreless.{0,15}first\b",
        r"\brun.{0,15}first inning\b", r"\b1st inning\b",
    ],
}

MARKET_TYPE_LABELS = {
    "moneyline":  "💵 Moneyline",
    "over_under": "📊 Over/Under",
    "run_line":   "📏 Run Line",
    "nrfi":       "🎯 NRFI/YRFI",
}


def classify_market(question: str) -> str:
    """Classify a market by its question text. Returns market type string."""
    q = question.lower()
    # Check in priority order (nrfi before moneyline since nrfi has 'win' logic)
    for mtype in ["nrfi", "run_line", "over_under", "moneyline"]:
        for pattern in MARKET_PATTERNS[mtype]:
            if re.search(pattern, q):
                return mtype
    return "other"


def match_market_to_game(market_question: str, game: dict) -> float:
    """
    Score how well a market matches a specific game (0.0 - 1.0).
    Uses team name keywords and date context.
    """
    q = market_question.lower()
    away_team = game.get("away_team", "")
    home_team = game.get("home_team", "")
    away_abbr = game.get("away_abbrev", "")
    home_abbr = game.get("home_abbrev", "")

    # Extract meaningful keywords from team names
    away_keywords = _team_keywords(away_team, away_abbr)
    home_keywords = _team_keywords(home_team, home_abbr)

    away_match = any(kw in q for kw in away_keywords)
    home_match = any(kw in q for kw in home_keywords)

    if away_match and home_match:
        return 1.0   # both teams mentioned — definite match
    elif away_match or home_match:
        return 0.5   # one team mentioned — possible match
    return 0.0


def _team_keywords(full_name: str, abbrev: str) -> list[str]:
    """Extract searchable keywords from a team name."""
    keywords = []
    if abbrev:
        keywords.append(abbrev.lower())
    if full_name:
        # Last word (e.g. "Yankees", "Dodgers")
        parts = full_name.lower().split()
        if parts:
            keywords.append(parts[-1])
        # Full name
        keywords.append(full_name.lower())
        # City only
        if len(parts) > 1:
            keywords.append(parts[0])
    return [k for k in keywords if len(k) >= 3]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════

async def scan_all_markets(
    games: list[dict],
    min_edge: float = 6.0,
    bankroll: float = 1000.0,
    kelly_frac: float = 0.5,
) -> dict:
    """
    Full sweep — for every game, find and evaluate all market types.

    Returns:
        {
          "edges":        list of all edges found (sorted by edge_pct),
          "value_plays":  list of value plays (model vs Vegas),
          "by_game":      dict keyed by game_id with per-game breakdown,
          "by_type":      dict keyed by market_type with all edges,
          "markets_found": int,
          "games_scanned": int,
        }
    """
    from heisenberg import find_mlb_markets, get_full_market_signal
    from polymarket import search_mlb_markets, get_market_implied_odds
    from alerts import _analyze_single_game
    from model_v2 import (
        find_edges_v2, get_best_value_plays,
        kelly_criterion, calculate_stake,
    )

    # ── Step 1: Pull ALL MLB markets from Heisenberg + Polymarket ────────
    log.info("Fetching all MLB markets...")
    market_sets = await asyncio.gather(
        find_mlb_markets("mlb", min_volume=0),
        find_mlb_markets("nrfi", min_volume=0),
        find_mlb_markets("run line", min_volume=0),
        find_mlb_markets("baseball over under", min_volume=0),
        search_mlb_markets("MLB"),
        return_exceptions=True,
    )

    # Deduplicate by condition_id / market id
    all_markets = {}
    for batch in market_sets:
        if isinstance(batch, Exception) or not batch:
            continue
        for m in batch:
            mid = m.get("condition_id") or m.get("id") or m.get("slug", "")
            if mid and mid not in all_markets:
                # Classify the market
                m["market_type"] = classify_market(m.get("question", ""))
                all_markets[mid] = m

    raw_markets = list(all_markets.values())
    log.info(f"Found {len(raw_markets)} unique MLB markets total")

    # ── Step 2: Match markets to games ────────────────────────────────────
    game_markets: dict[int, dict[str, list]] = {}  # game_id → {mtype: [markets]}

    for market in raw_markets:
        mtype = market.get("market_type", "other")
        if mtype == "other":
            continue

        best_score = 0.0
        best_game  = None

        for game in games:
            score = match_market_to_game(market.get("question", ""), game)
            if score > best_score:
                best_score = score
                best_game  = game

        if best_game and best_score >= 0.5:
            gid = best_game.get("game_id")
            if gid not in game_markets:
                game_markets[gid] = {
                    "moneyline": [], "over_under": [],
                    "run_line": [], "nrfi": [],
                }
            if mtype in game_markets[gid]:
                game_markets[gid][mtype].append(market)

    # ── Step 3: Run analysis + edge detection per game ────────────────────
    all_edges     = []
    all_value     = []
    by_game       = {}
    games_scanned = 0

    # Only scan pre-game / scheduled / live games
    scannable = [
        g for g in games
        if g.get("status") in ("Scheduled", "Pre-Game", "Warmup", "In Progress")
    ]

    for game in scannable:
        gid = game.get("game_id")
        matched = game_markets.get(gid, {})

        # Build flat list of matched markets for this game
        game_mkt_list = []
        for mtype, mlist in matched.items():
            for m in mlist:
                m["market_type"] = mtype
                game_mkt_list.append(m)

        if not game_mkt_list:
            log.debug(f"No markets matched for {game.get('away_abbrev')} @ {game.get('home_abbrev')}")

        try:
            analysis, edges, value_plays, heis_signals = await _analyze_single_game(
                game, game_mkt_list, min_edge
            )

            # Enrich edges with stake info
            for edge in edges:
                kelly = kelly_criterion(edge["model_prob"], edge["price"], kelly_frac)
                stake = calculate_stake(bankroll, kelly["kelly_pct"])
                edge["kelly"]  = kelly
                edge["stake"]  = stake
                edge["game"]   = game
                edge["analysis"] = analysis
                all_edges.append(edge)

            for vp in value_plays:
                vp["game"]     = game
                vp["analysis"] = analysis
                all_value.append(vp)

            # Per-game breakdown
            by_game[gid] = {
                "game":         game,
                "analysis":     analysis,
                "edges":        edges,
                "value_plays":  value_plays,
                "markets_by_type": {
                    mtype: len(mlist)
                    for mtype, mlist in matched.items()
                },
            }
            games_scanned += 1

        except Exception as e:
            log.warning(f"Scan failed for game {gid}: {e}")

    # ── Step 4: Sort and organise ─────────────────────────────────────────
    all_edges.sort(key=lambda e: e.get("edge_pct", 0), reverse=True)
    all_value.sort(key=lambda v: v.get("gap_pct", 0), reverse=True)

    # Group edges by market type
    by_type: dict[str, list] = {
        "moneyline": [], "over_under": [],
        "run_line": [], "nrfi": [],
    }
    for edge in all_edges:
        mt = edge.get("market_type", "other")
        if mt in by_type:
            by_type[mt].append(edge)

    return {
        "edges":          all_edges,
        "value_plays":    all_value,
        "by_game":        by_game,
        "by_type":        by_type,
        "markets_found":  len(raw_markets),
        "games_scanned":  games_scanned,
        "timestamp":      datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def format_full_scan_summary(result: dict) -> str:
    """Format the header summary of a full market scan."""
    edges      = result.get("edges", [])
    value      = result.get("value_plays", [])
    by_type    = result.get("by_type", {})
    n_markets  = result.get("markets_found", 0)
    n_games    = result.get("games_scanned", 0)

    # Count edges per type
    type_counts = {mt: len(elist) for mt, elist in by_type.items()}

    header = (
        f"{E['fire']} *Full Market Scan*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{E['calendar']} {datetime.now().strftime('%b %d, %Y  %I:%M %p')}\n\n"
        f"{E['chart']} Scanned `{n_markets}` markets across `{n_games}` games\n\n"
        f"*Edges found by type:*\n"
        f"  💵 Moneyline:  `{type_counts.get('moneyline', 0)}`\n"
        f"  📊 Over/Under: `{type_counts.get('over_under', 0)}`\n"
        f"  📏 Run Line:   `{type_counts.get('run_line', 0)}`\n"
        f"  🎯 NRFI/YRFI:  `{type_counts.get('nrfi', 0)}`\n"
        f"  {E['diamond']} Value vs Vegas: `{len(value)}`\n"
    )

    if not edges and not value:
        header += (
            f"\n{E['mag']} No edges found at current threshold.\n"
            f"Markets appear efficiently priced today.\n"
            f"Value plays vs Vegas shown below if available."
        )

    return header


def format_edge_by_type(by_type: dict, games_map: dict,
                         bankroll: float) -> list[str]:
    """
    Format edges grouped by market type.
    Returns a list of message strings (split to avoid Telegram 4096 char limit).
    """
    messages = []

    for mtype, label in MARKET_TYPE_LABELS.items():
        edges = by_type.get(mtype, [])
        if not edges:
            continue

        lines = [f"\n{label}\n{'─'*25}"]

        for edge in edges[:5]:  # top 5 per type
            game    = edge.get("game", {})
            kelly   = edge.get("kelly", {})
            stake   = edge.get("stake", 0)
            away    = game.get("away_abbrev", "???")
            home    = game.get("home_abbrev", "???")
            heis    = " ⚡" if edge.get("heis_confirms") else ""
            layers  = edge.get("layers_agreeing", [])
            layer_str = f" [{', '.join(layers)}]" if layers else ""

            lines.append(
                f"\n*{away} @ {home}*{heis}\n"
                f"  {E['target']} {edge.get('bet_desc', '')}\n"
                f"  Model: `{edge['model_prob']:.1%}` → Market: `{edge['market_prob']:.1%}`\n"
                f"  Edge: `+{edge['edge_pct']:.1f}%`  |  Stake: `${stake:.2f}`\n"
                f"  Kelly: `{kelly.get('kelly_pct', 0):.1f}%`  |  EV: `{kelly.get('ev_per_dollar', 0):.3f}`\n"
                f"  Conf: `{edge['confidence']:.0%}`{layer_str}"
            )

        messages.append("\n".join(lines))

    return messages


def format_value_plays_section(value_plays: list) -> str:
    """Format best value plays (model vs Vegas gaps)."""
    if not value_plays:
        return ""

    lines = [
        f"\n{E['diamond']} *Best Value — Model vs Sharp Books*\n"
        f"{'─'*25}\n"
        f"_No Polymarket edge needed — use these on_\n"
        f"_whichever platform has the better price._\n"
    ]

    for vp in value_plays[:6]:
        game = vp.get("game", {})
        away = game.get("away_abbrev", "???")
        home = game.get("home_abbrev", "???")
        vtype = "📊 vs Vegas" if vp.get("type") == "best_value" else "🧠 Model"

        lines.append(
            f"*{away} @ {home}*  {vtype}\n"
            f"  {E['target']} {vp.get('bet_desc', '')}\n"
            f"  Gap: `+{vp.get('gap_pct', 0):.1f}%`\n"
            f"  {vp.get('reasoning', '')}\n"
        )

    return "\n".join(lines)


def format_game_market_breakdown(game: dict, game_data: dict) -> str:
    """
    Format a single game's full market breakdown — all 4 types.
    Used when user taps a game to see its complete picture.
    """
    away     = game.get("away_abbrev", "???")
    home     = game.get("home_abbrev", "???")
    analysis = game_data.get("analysis", {})
    edges    = game_data.get("edges", [])
    value    = game_data.get("value_plays", [])
    mkt_counts = game_data.get("markets_by_type", {})

    comp = analysis.get("components", {})
    vegas = analysis.get("vegas", {})

    lines = [
        f"{E['base']} *{away} @ {home}* — Full Market View\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{E['brain']} *Model Probabilities*\n"
        f"```\n"
        f"{'Moneyline':<14} {away}: {analysis.get('away_win_prob', 0):.1%}  "
        f"{home}: {analysis.get('home_win_prob', 0):.1%}\n"
        f"{'Exp Total':<14} {analysis.get('expected_total', 0):.1f} runs\n"
        f"{'NRFI prob':<14} {analysis.get('fi_no_run_prob', 0):.1%}\n"
        f"{'YRFI prob':<14} {analysis.get('fi_run_prob', 0):.1%}\n"
        f"{'Run Env':<14} {analysis.get('run_environment', 1.0):.2f}x  "
        f"({analysis.get('weather_desc', 'N/A')})\n"
        f"```\n"
    ]

    # Vegas comparison
    if vegas.get("available"):
        lines.append(
            f"{E['money']} *Sharp Book Lines*\n"
            f"```\n"
            f"{'ML Home':<12} {vegas.get('home_implied', 0):.1%} "
            f"({vegas.get('home_dec_odds', 0):.2f}x)\n"
            f"{'ML Away':<12} {vegas.get('away_implied', 0):.1%} "
            f"({vegas.get('away_dec_odds', 0):.2f}x)\n"
            + (f"{'Total':<12} {vegas.get('consensus_total', 'N/A')}\n"
               if vegas.get("consensus_total") else "")
            + f"```\n"
        )

    # Markets found per type
    lines.append(f"{E['chart']} *Markets found:*\n")
    for mtype, label in MARKET_TYPE_LABELS.items():
        count = mkt_counts.get(mtype, 0)
        status = f"`{count} found`" if count else f"`none`"
        lines.append(f"  {label}: {status}")

    # Edges
    if edges:
        lines.append(f"\n{E['edge']} *{len(edges)} edge(s) detected:*")
        for edge in edges[:4]:
            heis = " ⚡" if edge.get("heis_confirms") else ""
            lines.append(
                f"  {MARKET_TYPE_LABELS.get(edge['market_type'], edge['market_type'])}"
                f"  {edge.get('bet_desc', '')}{heis}\n"
                f"    `+{edge['edge_pct']:.1f}%` edge  |  "
                f"Model `{edge['model_prob']:.1%}` vs Market `{edge['market_prob']:.1%}`"
            )
    elif value:
        lines.append(f"\n{E['diamond']} *No Poly edge — value vs Vegas:*")
        for vp in value[:2]:
            lines.append(
                f"  {vp.get('bet_desc', '')}  `+{vp.get('gap_pct', 0):.1f}%` gap"
            )
    else:
        lines.append(f"\n{E['mag']} No edges or value plays found for this game.")

    return "\n".join(lines)
