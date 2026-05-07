"""
market_scanner.py — Full-sweep Polymarket MLB market scanner.

Fixes:
  1. Only analyses PRE-GAME games — live/final games are flagged and skipped
     so stale model numbers never produce phantom edges
  2. Shows complete odds comparison table for ALL 4 market types per game
     regardless of whether an edge clears the threshold

For every pre-game on today's slate, finds and evaluates:
  1. Moneyline      (who wins)
  2. Over/Under     (total runs)
  3. Run Line       (spread ±1.5)
  4. NRFI/YRFI      (no/yes run first inning)
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from config import E

log = logging.getLogger("scanner")

# ── Game status buckets ───────────────────────────────────────────────────────
PREGAME_STATUSES = {
    "Scheduled", "Pre-Game", "Warmup", "Preview", "Delayed Start",
}
LIVE_STATUSES = {
    "In Progress", "Manager Challenge", "Delayed",
    "Rain Delay", "Suspended",
}
FINAL_STATUSES = {
    "Final", "Game Over", "Completed", "Postponed",
}

# ── Market type keyword patterns ──────────────────────────────────────────────
MARKET_PATTERNS = {
    "nrfi": [
        r"\bnrfi\b", r"\byrfi\b", r"\bno run.{0,20}first\b",
        r"\bfirst inning\b", r"\bscoreless.{0,20}first\b",
        r"\brun.{0,20}first inning\b", r"\b1st inning\b",
    ],
    "run_line": [
        r"\brun line\b", r"\bspread\b", r"\b[+-]1\.5\b",
        r"\bhandicap\b", r"\bcover\b",
    ],
    "over_under": [
        r"\bover\b", r"\bunder\b", r"\btotal runs?\b",
        r"\bcombined runs?\b", r"\bmore than\b", r"\bfewer than\b",
    ],
    "moneyline": [
        r"\bwin\b", r"\bwinner\b", r"\bbeat\b", r"\bdefeats?\b",
        r"\bmoneyline\b",
    ],
}

MARKET_TYPE_LABELS = {
    "moneyline":  "💵 Moneyline",
    "over_under": "📊 Over/Under",
    "run_line":   "📏 Run Line",
    "nrfi":       "🎯 NRFI/YRFI",
}


# ─── helpers ────────────────────────────────────────────────────────────────

def classify_market(question: str) -> str:
    q = question.lower()
    for mtype in ["nrfi", "run_line", "over_under", "moneyline"]:
        for pattern in MARKET_PATTERNS[mtype]:
            if re.search(pattern, q):
                return mtype
    return "other"


def game_status_category(status: str) -> str:
    if status in PREGAME_STATUSES:   return "pregame"
    if status in LIVE_STATUSES:      return "live"
    if status in FINAL_STATUSES:     return "final"
    s = status.lower()
    if any(w in s for w in ("progress", "live", "delay", "rain")): return "live"
    if any(w in s for w in ("final", "over", "complete")):          return "final"
    if any(w in s for w in ("scheduled", "pre", "warmup")):         return "pregame"
    return "unknown"


def match_score(question: str, game: dict) -> float:
    q = question.lower()
    away_kws = _keywords(game.get("away_team", ""), game.get("away_abbrev", ""))
    home_kws = _keywords(game.get("home_team", ""), game.get("home_abbrev", ""))
    away_hit = any(kw in q for kw in away_kws)
    home_hit = any(kw in q for kw in home_kws)
    if away_hit and home_hit: return 1.0
    if away_hit or home_hit:  return 0.5
    return 0.0


def _keywords(full_name: str, abbrev: str) -> list:
    kws = []
    if abbrev:
        kws.append(abbrev.lower())
    if full_name:
        parts = full_name.lower().split()
        if parts: kws.append(parts[-1])
        kws.append(full_name.lower())
    return [k for k in kws if len(k) >= 3]


def _match_outcome_to_model(name: str, mtype: str, analysis: dict) -> Optional[float]:
    n = name.lower()
    if mtype == "moneyline":
        if any(k in n for k in ["home", "yes"]): return analysis.get("home_win_prob")
        if any(k in n for k in ["away", "no"]):  return analysis.get("away_win_prob")
    elif mtype == "nrfi":
        if any(k in n for k in ["no", "nrfi"]):  return analysis.get("fi_no_run_prob")
        if any(k in n for k in ["yes", "yrfi"]): return analysis.get("fi_run_prob")
    elif mtype == "over_under":
        exp = analysis.get("expected_total", 0)
        m = re.search(r"(\d+\.?\d*)", n)
        if m:
            from model_v2 import _over_probability
            line = float(m.group(1))
            if "over"  in n: return _over_probability(exp, line)
            if "under" in n: return 1 - _over_probability(exp, line)
    elif mtype == "run_line":
        hp = analysis.get("home_win_prob", 0.5)
        ap = analysis.get("away_win_prob", 0.5)
        if any(k in n for k in ["home", "-1.5 home"]):
            return hp * 0.72 if hp > 0.5 else hp * 0.60
        if any(k in n for k in ["away", "-1.5 away"]):
            return ap * 0.72 if ap > 0.5 else ap * 0.60
        if "+1.5" in n:
            return 1 - (max(hp, ap) * 0.72)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_all_mlb_markets(games: list) -> list:
    """Aggressively fetch all MLB markets using generic + per-team searches."""
    from heisenberg import find_mlb_markets
    from polymarket import search_mlb_markets

    terms = ["mlb", "nrfi", "yrfi", "baseball"]
    for g in games:
        for side in ["away_team", "home_team"]:
            last = (g.get(side) or "").split()[-1].lower()
            if last and len(last) >= 4:
                terms.append(last)
    terms = list(dict.fromkeys(terms))[:15]

    tasks = [find_mlb_markets(t, min_volume=0) for t in terms]
    tasks.append(search_mlb_markets("MLB"))
    batches = await asyncio.gather(*tasks, return_exceptions=True)

    seen: dict = {}
    for batch in batches:
        if isinstance(batch, Exception) or not batch:
            continue
        for m in batch:
            mid = (m.get("condition_id") or m.get("id") or
                   m.get("slug") or m.get("question", ""))
            if mid and mid not in seen:
                m["market_type"] = classify_market(m.get("question", ""))
                seen[mid] = m

    return [m for m in seen.values() if m.get("market_type") != "other"]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════

async def scan_all_markets(
    games: list,
    min_edge: float = 6.0,
    bankroll: float = 1000.0,
    kelly_frac: float = 0.5,
) -> dict:
    """
    Full sweep — pre-game games only, all 4 market types, complete comparison table.
    """
    from polymarket import get_market_implied_odds
    from alerts import _analyze_single_game
    from model_v2 import kelly_criterion, calculate_stake

    # ── Split games by status ─────────────────────────────────────────────
    pregame  = []
    skipped  = []
    for g in games:
        cat = game_status_category(g.get("status", ""))
        if cat == "pregame":
            pregame.append(g)
        else:
            skipped.append({"game": g, "reason": cat})

    if not pregame:
        return {
            "edges": [], "value_plays": [], "by_game": {},
            "by_type": {k: [] for k in MARKET_TYPE_LABELS},
            "markets_found": 0, "games_scanned": 0,
            "skipped": skipped,
            "message": "No pre-game games. All games are live or final.",
        }

    # ── Fetch all markets once ────────────────────────────────────────────
    all_markets = await fetch_all_mlb_markets(pregame)

    # ── Match markets → games ─────────────────────────────────────────────
    game_markets: dict = {}
    for market in all_markets:
        mtype      = market.get("market_type", "other")
        best_score = 0.0
        best_game  = None
        for game in pregame:
            sc = match_score(market.get("question", ""), game)
            if sc > best_score:
                best_score = sc
                best_game  = game
        if best_game and best_score >= 0.5:
            gid = best_game.get("game_id")
            if gid not in game_markets:
                game_markets[gid] = {k: [] for k in MARKET_TYPE_LABELS}
            if mtype in game_markets[gid]:
                game_markets[gid][mtype].append(market)

    # ── Fetch implied odds for all matched markets ────────────────────────
    refs, tasks = [], []
    for gid, tdict in game_markets.items():
        for mtype, mlist in tdict.items():
            for m in mlist:
                tasks.append(get_market_implied_odds(m))
                refs.append((gid, mtype, m))

    if tasks:
        odds_results = await asyncio.gather(*tasks, return_exceptions=True)
        for (gid, mtype, m), odds in zip(refs, odds_results):
            if not isinstance(odds, Exception) and odds:
                m["outcomes_with_odds"] = odds.get("outcomes", m.get("outcomes", []))
            else:
                m["outcomes_with_odds"] = m.get("outcomes", [])
            m["market_type"] = mtype

    # ── Analyse each game ─────────────────────────────────────────────────
    all_edges  = []
    all_value  = []
    by_game    = {}
    n_scanned  = 0

    for game in pregame:
        gid      = game.get("game_id")
        matched  = game_markets.get(gid, {k: [] for k in MARKET_TYPE_LABELS})
        flat_mkts = [m for ml in matched.values() for m in ml]

        try:
            analysis, edges, value_plays, _ = await _analyze_single_game(
                game, flat_mkts, min_edge
            )

            for edge in edges:
                kelly = kelly_criterion(edge["model_prob"], edge["price"], kelly_frac)
                stake = calculate_stake(bankroll, kelly["kelly_pct"])
                edge.update({"kelly": kelly, "stake": stake,
                              "game": game, "analysis": analysis})
                all_edges.append(edge)

            for vp in value_plays:
                vp.update({"game": game, "analysis": analysis})
                all_value.append(vp)

            by_game[gid] = {
                "game":        game,
                "analysis":    analysis,
                "edges":       edges,
                "value_plays": value_plays,
                "comparison":  _build_comparison(analysis, matched, min_edge),
                "markets_by_type": {mt: len(ml) for mt, ml in matched.items()},
            }
            n_scanned += 1

        except Exception as e:
            log.warning(f"Analysis failed game {gid}: {e}")
            by_game[gid] = {
                "game": game, "edges": [], "value_plays": [],
                "comparison": {}, "error": str(e),
            }

    all_edges.sort(key=lambda e: e.get("edge_pct", 0), reverse=True)
    all_value.sort(key=lambda v: v.get("gap_pct", 0),  reverse=True)

    by_type = {k: [] for k in MARKET_TYPE_LABELS}
    for edge in all_edges:
        mt = edge.get("market_type", "")
        if mt in by_type:
            by_type[mt].append(edge)

    return {
        "edges":         all_edges,
        "value_plays":   all_value,
        "by_game":       by_game,
        "by_type":       by_type,
        "markets_found": len(all_markets),
        "games_scanned": n_scanned,
        "skipped":       skipped,
        "timestamp":     datetime.utcnow().isoformat(),
    }


def _build_comparison(analysis: dict, matched: dict, min_edge: float) -> dict:
    """Build full model-vs-market comparison for every type."""
    table = {}
    for mtype in MARKET_TYPE_LABELS:
        markets = matched.get(mtype, [])
        table[mtype] = {"markets": []}
        for mkt in markets:
            outcomes = mkt.get("outcomes_with_odds", mkt.get("outcomes", []))
            rows = []
            for o in outcomes:
                name       = o.get("name", "")
                mkt_p      = o.get("implied_prob") or o.get("price")
                try:   mkt_p = float(mkt_p) if mkt_p is not None else None
                except: mkt_p = None
                mdl_p  = _match_outcome_to_model(name, mtype, analysis)
                edge_p = round((mdl_p - mkt_p) * 100, 1) if mdl_p and mkt_p else None
                rows.append({
                    "name":       name,
                    "market_prob": mkt_p,
                    "model_prob":  mdl_p,
                    "edge_pct":   edge_p,
                    "has_edge":   edge_p is not None and edge_p >= min_edge,
                })
            table[mtype]["markets"].append({
                "question": mkt.get("question", ""),
                "volume":   mkt.get("volume", 0),
                "outcomes": rows,
            })
    return table


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def format_full_scan_summary(result: dict) -> str:
    by_type   = result.get("by_type", {})
    value     = result.get("value_plays", [])
    skipped   = result.get("skipped", [])
    n_markets = result.get("markets_found", 0)
    n_scanned = result.get("games_scanned", 0)
    message   = result.get("message", "")

    live_ct  = sum(1 for s in skipped if s["reason"] == "live")
    final_ct = sum(1 for s in skipped if s["reason"] == "final")

    lines = [
        f"🔭 *Full Market Scan*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{E['calendar']} {datetime.now().strftime('%b %d  %I:%M %p')}\n\n"
        f"📋 Pre-game scanned: `{n_scanned}`  |  "
        f"Markets found: `{n_markets}`\n"
    ]
    if live_ct:
        lines.append(
            f"  {E['green']} `{live_ct}` game(s) live — "
            f"*skipped* (pre-game model only)\n"
        )
    if final_ct:
        lines.append(f"  {E['check']} `{final_ct}` game(s) final — skipped\n")

    lines.append("\n*Edges vs Polymarket:*")
    for mtype, label in MARKET_TYPE_LABELS.items():
        n = len(by_type.get(mtype, []))
        lines.append(f"  {label}:  `{n}`")
    lines.append(f"  {E['diamond']} Value vs Vegas: `{len(value)}`")

    if message:
        lines.append(f"\n{E['yellow']} _{message}_")
    elif not result.get("edges") and not value:
        lines.append(
            f"\n{E['mag']} No edges above threshold.\n"
            f"Tap any game for full model vs market table."
        )
    return "\n".join(lines)


def format_game_full_comparison(game_data: dict) -> list:
    """
    Returns a list of message strings for one game —
    full model vs market table for all 4 types.
    """
    game       = game_data.get("game", {})
    analysis   = game_data.get("analysis", {}) or {}
    comparison = game_data.get("comparison", {}) or {}
    edges      = game_data.get("edges", [])
    value      = game_data.get("value_plays", [])
    vegas      = analysis.get("vegas", {}) or {}

    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")
    conf = analysis.get("confidence", 0)
    conf_e = E["green"] if conf >= 0.6 else E["yellow"] if conf >= 0.35 else E["red"]

    messages = []

    # ── Header ────────────────────────────────────────────────────────────
    hdr = [
        f"{E['base']} *{away} @ {home}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{conf_e} Confidence: `{conf:.0%}`"
    ]
    wd = analysis.get("weather_desc", "")
    re_v = analysis.get("run_environment", 1.0)
    if wd and wd not in ("Neutral", "Dome/indoor", "Weather unavailable"):
        arrow = "⬆️" if re_v > 1.01 else "⬇️"
        hdr.append(f"🌤 {wd}  {arrow} run env `{re_v:.2f}x`")
    if vegas.get("available"):
        hdr.append(
            f"{E['money']} Vegas ML: `{away} {vegas.get('away_implied',0):.1%}` / "
            f"`{home} {vegas.get('home_implied',0):.1%}`"
            + (f"  Total: `{vegas['consensus_total']}`" if vegas.get("consensus_total") else "")
        )
    messages.append("\n".join(hdr))

    # ── One message per market type ───────────────────────────────────────
    for mtype, label in MARKET_TYPE_LABELS.items():
        data    = comparison.get(mtype, {})
        markets = data.get("markets", [])
        lines   = [f"{label}"]

        if not markets:
            # No market found — show model numbers anyway
            lines.append(_no_market_model_line(mtype, analysis, away, home))
        else:
            for mkt in markets[:2]:
                q   = (mkt.get("question") or "")[:55]
                vol = mkt.get("volume", 0)
                vol_str = f"  Vol `${vol:,.0f}`" if vol else ""
                lines.append(f"`{q}`{vol_str}")

                # Table header
                lines.append(f"```")
                lines.append(f"{'Outcome':<17} {'Mkt':>6} {'Model':>6} {'Edge':>7}")
                lines.append(f"{'─'*38}")

                for row in mkt.get("outcomes", []):
                    name  = (row.get("name") or "")[:16]
                    mp    = row.get("market_prob")
                    mdl   = row.get("model_prob")
                    ep    = row.get("edge_pct")
                    flag  = " <" if row.get("has_edge") else ""

                    mp_s  = f"{mp:.1%}"  if mp  is not None else " N/A "
                    mdl_s = f"{mdl:.1%}" if mdl is not None else " N/A "
                    ep_s  = f"{ep:+.1f}%" if ep  is not None else "  N/A "

                    lines.append(
                        f"{name:<17} {mp_s:>6} {mdl_s:>6} {ep_s:>7}{flag}"
                    )
                lines.append(f"```")

        messages.append("\n".join(lines))

    # ── Best plays summary ────────────────────────────────────────────────
    if edges or value:
        pl = [f"{E['edge']} *Best plays:*"]
        for edge in edges[:3]:
            heis = " ⚡" if edge.get("heis_confirms") else ""
            kelly = edge.get("kelly", {})
            pl.append(
                f"  {MARKET_TYPE_LABELS.get(edge['market_type'], '')} "
                f"*{edge.get('bet_desc', '')}*{heis}\n"
                f"    `+{edge['edge_pct']:.1f}%` | "
                f"Stake `${edge.get('stake', 0):.2f}` | "
                f"Kelly `{kelly.get('kelly_pct', 0):.1f}%`"
            )
        for vp in value[:2]:
            if not edges:
                pl.append(
                    f"  {E['diamond']} {vp.get('bet_desc','')}"
                    f"  `+{vp.get('gap_pct',0):.1f}%` vs Vegas"
                )
        messages.append("\n".join(pl))

    return messages


def _no_market_model_line(mtype: str, analysis: dict,
                           away: str, home: str) -> str:
    if mtype == "moneyline":
        hw = analysis.get("home_win_prob", 0)
        aw = analysis.get("away_win_prob", 0)
        return (f"  Model: `{away}` `{aw:.1%}` | `{home}` `{hw:.1%}`\n"
                f"  _No Polymarket line found_")
    elif mtype == "nrfi":
        nrfi = analysis.get("fi_no_run_prob", 0)
        yrfi = analysis.get("fi_run_prob", 0)
        return f"  Model: NRFI `{nrfi:.1%}` | YRFI `{yrfi:.1%}`"
    elif mtype == "over_under":
        exp = analysis.get("expected_total", 0)
        vt  = (analysis.get("vegas", {}) or {}).get("consensus_total")
        return (f"  Model total: `{exp:.1f}` runs"
                + (f"  Vegas line: `{vt}`" if vt else ""))
    elif mtype == "run_line":
        hw = analysis.get("home_win_prob", 0.5)
        aw = analysis.get("away_win_prob", 0.5)
        return (f"  Approx: `{home} -1.5` ≈ `{hw*0.72:.1%}` | "
                f"`{away} -1.5` ≈ `{aw*0.72:.1%}`")
    return "  _No data_"


def format_full_scan_edges_by_type(by_type: dict, bankroll: float) -> list:
    messages = []
    for mtype, label in MARKET_TYPE_LABELS.items():
        edges = by_type.get(mtype, [])
        if not edges:
            continue
        lines = [f"{label} — *{len(edges)} edge(s)*\n{'─'*28}"]
        for edge in edges[:6]:
            game  = edge.get("game", {})
            kelly = edge.get("kelly", {})
            away  = game.get("away_abbrev", "???")
            home  = game.get("home_abbrev", "???")
            heis  = " ⚡" if edge.get("heis_confirms") else ""
            layers = edge.get("layers_agreeing", [])
            lines.append(
                f"\n*{away} @ {home}*{heis}\n"
                f"  {edge.get('bet_desc','')}\n"
                f"  Mkt `{edge['market_prob']:.1%}` → "
                f"Model `{edge['model_prob']:.1%}`\n"
                f"  Edge `+{edge['edge_pct']:.1f}%` | "
                f"Stake `${edge.get('stake',0):.2f}` | "
                f"Kelly `{kelly.get('kelly_pct',0):.1f}%`"
                + (f"\n  [{', '.join(layers)}]" if layers else "")
            )
        messages.append("\n".join(lines))
    return messages


def format_value_plays_section(value_plays: list) -> str:
    if not value_plays:
        return ""
    lines = [
        f"{E['diamond']} *Value vs Sharp Books*\n"
        f"{'─'*28}\n"
        f"_Model vs Vegas disagrees ≥3%. Find the better price._\n"
    ]
    for vp in value_plays[:6]:
        game = vp.get("game", {})
        away = game.get("away_abbrev", "???")
        home = game.get("home_abbrev", "???")
        vtype = "📊 vs Vegas" if vp.get("type") == "best_value" else "🧠 Model"
        lines.append(
            f"*{away} @ {home}*  {vtype}\n"
            f"  {vp.get('bet_desc','')}\n"
            f"  Gap `+{vp.get('gap_pct',0):.1f}%`  "
            f"Model `{vp.get('model_prob',0):.1%}`\n"
            f"  _{vp.get('reasoning','')}_\n"
        )
    return "\n".join(lines)


def format_skipped_games(skipped: list) -> str:
    if not skipped:
        return ""
    live  = [s for s in skipped if s["reason"] == "live"]
    final = [s for s in skipped if s["reason"] == "final"]
    lines = []
    if live:
        lines.append(f"\n{E['green']} *Live (skipped from edge model):*")
        for s in live:
            g = s["game"]
            inn  = g.get("inning", "?")
            half = "▲" if "Top" in g.get("inning_state", "") else "▼"
            lines.append(
                f"  `{g.get('away_abbrev')}` "
                f"{g.get('away_score',0)}-{g.get('home_score',0)} "
                f"`{g.get('home_abbrev')}` {half}{inn}"
            )
    if final:
        lines.append(f"\n{E['check']} *Final:*")
        for s in final:
            g = s["game"]
            lines.append(
                f"  `{g.get('away_abbrev')}` "
                f"{g.get('away_score',0)}-{g.get('home_score',0)} "
                f"`{g.get('home_abbrev')}` F"
            )
    return "\n".join(lines)
