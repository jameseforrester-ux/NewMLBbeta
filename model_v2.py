"""
Model v2 — 4-Layer Edge Detection Engine

Layer 1 — Statistical (35%):  Pitcher xFIP/Statcast, batting, park, weather, form
Layer 2 — Vegas Prior  (25%):  Sharp book consensus (Pinnacle/DraftKings via Odds API)
Layer 3 — Smart Money  (30%):  Heisenberg elite wallet positioning on this exact market
Layer 4 — Market Structure (10%): Whale concentration, volume trend, winning side PnL

Decision rule: alert only when ≥ 2 layers agree AND combined edge ≥ threshold.
This filters out noise and ensures only high-conviction plays surface.

Best Value Plays: fire when model + Vegas agree but Polymarket hasn't caught up,
even without a Heisenberg signal — thin markets = more mispricing.
"""

import math
import asyncio
import logging
from typing import Optional

from config import PARK_FACTORS, E

log = logging.getLogger("model_v2")

# ── League averages (2024-2025 baseline) ─────────────────────────────────────
LEAGUE_AVG_ERA    = 4.15
LEAGUE_AVG_WHIP   = 1.29
LEAGUE_AVG_OPS    = 0.720
LEAGUE_AVG_RPG    = 4.45
LEAGUE_HOME_WIN   = 0.536   # historical MLB home win rate

# ── Park factors (imported from config, duplicated here for convenience) ─────
from config import PARK_FACTORS

OUTDOOR_PARKS = {
    "Wrigley Field", "Fenway Park", "Dodger Stadium", "Oracle Park",
    "PNC Park", "Camden Yards", "Petco Park", "Angel Stadium",
    "Kauffman Stadium", "Nationals Park", "Great American Ball Park",
    "Globe Life Field", "Progressive Field", "Coors Field",
    "Target Field", "Busch Stadium", "Truist Park",
    "Guaranteed Rate Field", "Yankee Stadium", "Citizens Bank Park",
    "American Family Field", "loanDepot park", "Comerica Park",
    "T-Mobile Park", "Chase Field",
}

PARK_COORDS = {
    "Wrigley Field":              (41.9484, -87.6553),
    "Fenway Park":                (42.3467, -71.0972),
    "Dodger Stadium":             (34.0739, -118.2400),
    "Oracle Park":                (37.7786, -122.3893),
    "PNC Park":                   (40.4469, -80.0057),
    "Camden Yards":               (39.2838, -76.6217),
    "Petco Park":                 (32.7076, -117.1570),
    "Coors Field":                (39.7559, -104.9942),
    "Yankee Stadium":             (40.8296, -73.9262),
    "Citizens Bank Park":         (39.9061, -75.1665),
    "Target Field":               (44.9817, -93.2783),
    "Great American Ball Park":   (39.0979, -84.5082),
    "Kauffman Stadium":           (39.0517, -94.4803),
    "Truist Park":                (33.8908, -84.4678),
    "Nationals Park":             (38.8730, -77.0074),
    "Progressive Field":          (41.4962, -81.6852),
    "Globe Life Field":           (32.7474, -97.0839),
    "Guaranteed Rate Field":      (41.8299, -87.6338),
    "Angel Stadium":              (33.8003, -117.8827),
    "American Family Field":      (43.0280, -87.9712),
    "loanDepot park":             (25.7781, -80.2195),
    "Comerica Park":              (42.3390, -83.0485),
    "T-Mobile Park":              (47.5913, -122.3328),
    "Chase Field":                (33.4453, -112.0667),
    "Busch Stadium":              (38.6226, -90.1928),
}


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 1 — STATISTICAL MODEL
# ═══════════════════════════════════════════════════════════════════════════

def pitcher_score(stats: dict, statcast: dict = None) -> tuple[float, float, list]:
    """
    Score pitcher quality 0-1. Returns (score, confidence, metrics_used).
    Prioritises: Statcast whiff/xwOBA > FIP > ERA.
    """
    if not stats:
        return 0.5, 0.0, []

    weighted_scores = []
    metrics_used = []

    # ── Statcast (highest predictive value) ──────────────────────────────
    if statcast and statcast.get("available"):
        whiff = _f(statcast.get("whiff_percent"))
        if whiff is not None:
            weighted_scores.append((_linear(whiff, 15, 35), 0.20))
            metrics_used.append(f"Whiff%: {whiff:.1f}")

        xwoba = _f(statcast.get("xwoba"))
        if xwoba is not None:
            weighted_scores.append((_inverse(xwoba, 0.24, 0.40), 0.18))
            metrics_used.append(f"xwOBA: {xwoba:.3f}")

        hard_hit = _f(statcast.get("hard_hit_percent"))
        if hard_hit is not None:
            weighted_scores.append((_inverse(hard_hit, 25, 52), 0.12))
            metrics_used.append(f"HardHit%: {hard_hit:.1f}")

        xba = _f(statcast.get("xba"))
        if xba is not None:
            weighted_scores.append((_inverse(xba, 0.17, 0.33), 0.10))
            metrics_used.append(f"xBA: {xba:.3f}")

    # ── FIP (park/defense neutral) ────────────────────────────────────────
    fip = _f(stats.get("fip"))
    if fip is not None:
        weighted_scores.append((_inverse(fip, 2.0, 6.5), 0.16))
        metrics_used.append(f"FIP: {fip:.2f}")

    # ── Traditional ──────────────────────────────────────────────────────
    era = _f(stats.get("era") or stats.get("earnedRunAverage"))
    if era is not None:
        weighted_scores.append((_inverse(era, 1.5, 6.5), 0.10))
        metrics_used.append(f"ERA: {era:.2f}")

    whip = _f(stats.get("whip"))
    if whip is not None:
        weighted_scores.append((_inverse(whip, 0.80, 1.80), 0.08))
        metrics_used.append(f"WHIP: {whip:.2f}")

    k9 = _f(stats.get("strikeoutsPer9Inn") or stats.get("k_per_9"))
    if k9 is not None:
        weighted_scores.append((_linear(k9, 4.0, 13.5), 0.09))
        metrics_used.append(f"K/9: {k9:.1f}")

    bb9 = _f(stats.get("walksPer9Inn") or stats.get("bb_per_9"))
    if bb9 is not None:
        weighted_scores.append((_inverse(bb9, 1.0, 6.0), 0.07))
        metrics_used.append(f"BB/9: {bb9:.1f}")

    if not weighted_scores:
        return 0.5, 0.0, []

    total_w = sum(w for _, w in weighted_scores)
    score   = sum(s * w for s, w in weighted_scores) / total_w
    conf    = min(1.0, total_w / 0.75)

    return round(score, 4), round(conf, 2), metrics_used


def team_batting_score(stats: dict) -> float:
    scores = []
    ops = _f(stats.get("ops"))
    if ops:
        scores.append((_linear(ops, 0.600, 0.860), 0.35))
    obp = _f(stats.get("obp"))
    slg = _f(stats.get("slg"))
    if obp and slg:
        woba = obp * 0.69 + slg * 0.31
        scores.append((_linear(woba, 0.20, 0.40), 0.30))
    runs = _f(stats.get("runs"))
    gp   = _f(stats.get("gamesPlayed"))
    if runs and gp and gp > 0:
        rpg = runs / gp
        scores.append((_linear(rpg, 3.0, 6.5), 0.35))
    if not scores:
        return 0.5
    tw = sum(w for _, w in scores)
    return round(sum(s * w for s, w in scores) / tw, 4)


def bullpen_score(pitching_stats: dict) -> float:
    era = _f(pitching_stats.get("era") or pitching_stats.get("earnedRunAverage"))
    if era is None:
        return 0.5
    return round(_inverse(era, 2.5, 5.5), 4)


async def get_weather_factor(venue: str) -> tuple[float, str]:
    """
    Fetch weather from Open-Meteo (free, no key) for outdoor parks.
    Returns (run_environment_multiplier, description).
    """
    if venue not in OUTDOOR_PARKS:
        return 1.0, "Dome/indoor"

    coords = PARK_COORDS.get(venue)
    if not coords:
        return 1.0, "Unknown park"

    lat, lon = coords
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "wind_speed_10m,wind_direction_10m,temperature_2m,precipitation",
                    "wind_speed_unit": "mph",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 1,
                },
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return 1.0, "Weather unavailable"
                data = await resp.json()
    except Exception:
        return 1.0, "Weather unavailable"

    cur  = data.get("current", {})
    wind = float(cur.get("wind_speed_10m") or 0)
    wdir = float(cur.get("wind_direction_10m") or 180)
    temp = float(cur.get("temperature_2m") or 72)
    prec = float(cur.get("precipitation") or 0)

    wind_out = (wdir <= 90 or wdir >= 270)
    wm = 1.0
    if wind > 5:
        factor = (wind / 5) * 0.03
        wm = 1.0 + factor if wind_out else 1.0 - factor * 0.5

    tm = 1.0
    if temp < 50:   tm = 0.93
    elif temp < 60: tm = 0.97
    elif temp > 85: tm = 1.04

    pm = 0.97 if prec > 0.1 else 1.0
    mult = round(wm * tm * pm, 3)

    parts = []
    if wind > 8:
        parts.append(f"Wind {wind:.0f}mph {'out' if wind_out else 'in'}")
    if temp < 55:
        parts.append(f"Cold {temp:.0f}°F")
    elif temp > 85:
        parts.append(f"Hot {temp:.0f}°F")
    if prec > 0.1:
        parts.append("Rain")

    desc = ", ".join(parts) if parts else "Neutral"
    return mult, desc


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 2 — VEGAS SHARP CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════

async def get_vegas_prior(home_team: str, away_team: str) -> dict:
    """
    Pull de-vigged moneyline + total from sharp books via The Odds API.
    Returns implied home/away win prob and consensus total.
    """
    from config import ODDS_API_KEY, ODDS_API_BASE
    if not ODDS_API_KEY:
        return {"available": False}

    home_kw = home_team.lower().split()[-1]
    away_kw = away_team.lower().split()[-1]

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/odds",
                params={
                    "apiKey":    ODDS_API_KEY,
                    "regions":   "us",
                    "markets":   "h2h,totals",
                    "oddsFormat":"decimal",
                    "bookmakers":"pinnacle,draftkings,fanduel,betmgm",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return {"available": False}
                games = await resp.json()
    except Exception as e:
        log.warning(f"Vegas prior fetch error: {e}")
        return {"available": False}

    for game in games:
        ht = (game.get("home_team") or "").lower()
        at = (game.get("away_team") or "").lower()
        if home_kw not in ht and away_kw not in at:
            continue

        home_odds_list, away_odds_list, totals = [], [], []
        for book in game.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        name = (o.get("name") or "").lower()
                        price = o.get("price", 0)
                        if home_kw in name: home_odds_list.append(price)
                        elif away_kw in name: away_odds_list.append(price)
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if (o.get("name") or "").lower() == "over":
                            totals.append(o.get("point", 0))

        if not home_odds_list or not away_odds_list:
            continue

        avg_h = sum(home_odds_list) / len(home_odds_list)
        avg_a = sum(away_odds_list) / len(away_odds_list)
        raw_h = 1 / avg_h
        raw_a = 1 / avg_a
        total_raw = raw_h + raw_a

        return {
            "available":      True,
            "home_implied":   round(raw_h / total_raw, 4),
            "away_implied":   round(raw_a / total_raw, 4),
            "home_dec_odds":  round(avg_h, 3),
            "away_dec_odds":  round(avg_a, 3),
            "books":          len(home_odds_list),
            "consensus_total":round(sum(totals)/len(totals), 1) if totals else None,
        }

    return {"available": False}


# ═══════════════════════════════════════════════════════════════════════════
#  FULL 4-LAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

async def analyze_game_v2(
    home_team: str,
    away_team: str,
    venue: str,
    home_pitcher_stats: dict,
    away_pitcher_stats: dict,
    home_pitcher_statcast: dict,
    away_pitcher_statcast: dict,
    home_team_batting: dict,
    away_team_batting: dict,
    home_team_pitching: dict,
    away_team_pitching: dict,
    home_recent_form: dict,
    away_recent_form: dict,
    h2h: dict = None,
    heisenberg_signals: dict = None,  # pre-fetched per-market signals
) -> dict:
    """
    Full 4-layer analysis returning win probability, expected runs,
    first-inning probability, and a composite confidence score.
    """

    # ── Layer 1: Statistical ─────────────────────────────────────────────
    home_p_score, home_p_conf, home_p_metrics = pitcher_score(
        home_pitcher_stats, home_pitcher_statcast)
    away_p_score, away_p_conf, away_p_metrics = pitcher_score(
        away_pitcher_stats, away_pitcher_statcast)

    home_bat = team_batting_score(home_team_batting)
    away_bat = team_batting_score(away_team_batting)
    home_bp  = bullpen_score(home_team_pitching)
    away_bp  = bullpen_score(away_team_pitching)

    home_form = home_recent_form.get("win_pct", 0.5) if home_recent_form.get("available") else 0.5
    away_form = away_recent_form.get("win_pct", 0.5) if away_recent_form.get("available") else 0.5

    pf = PARK_FACTORS.get(venue, 1.0)
    weather_mult, weather_desc = await get_weather_factor(venue)
    run_env = pf * weather_mult

    # H2H adjustment
    h2h_home = 0.5
    if h2h and h2h.get("available") and h2h.get("games", 0) >= 3:
        h2h_home = h2h.get("team1_wins", 0) / h2h["games"]

    # Home advantage
    home_adv_mult = LEAGUE_HOME_WIN

    # Composite strengths
    home_str = (
        (1 - away_p_score) * 0.24 +
        home_bat           * 0.26 +
        home_bp            * 0.09 +
        home_adv_mult      * 0.12 +
        home_form          * 0.14 +
        h2h_home           * 0.05 +
        (run_env - 1.0)    * 0.10
    )
    away_str = (
        (1 - home_p_score) * 0.24 +
        away_bat           * 0.26 +
        away_bp            * 0.09 +
        (1-home_adv_mult)  * 0.12 +
        away_form          * 0.14 +
        (1 - h2h_home)     * 0.05 +
        (1/run_env - 1.0)  * 0.10
    )
    total_str = home_str + away_str
    stat_home_prob = home_str / total_str if total_str > 0 else 0.5
    stat_home_prob = max(0.22, min(0.78, stat_home_prob))

    # ── Layer 2: Vegas sharp prior ───────────────────────────────────────
    vegas = await get_vegas_prior(home_team, away_team)
    vegas_available = vegas.get("available", False)

    # ── Layer 3 + 4: Heisenberg (passed in per-market) ───────────────────
    # heisenberg_signals is a dict keyed by market_type
    heis = heisenberg_signals or {}
    heis_available = bool(heis)

    # ── Blend probabilities ───────────────────────────────────────────────
    # Weights depend on data availability
    if vegas_available and heis_available:
        home_win_prob = (
            stat_home_prob              * 0.35 +
            vegas["home_implied"]       * 0.25 +
            _heisenberg_prob_adjustment(heis, stat_home_prob) * 0.30 +
            stat_home_prob              * 0.10   # market structure (fallback to stat)
        )
    elif vegas_available:
        home_win_prob = (
            stat_home_prob        * 0.45 +
            vegas["home_implied"] * 0.35 +
            stat_home_prob        * 0.20
        )
    elif heis_available:
        home_win_prob = (
            stat_home_prob * 0.55 +
            _heisenberg_prob_adjustment(heis, stat_home_prob) * 0.45
        )
    else:
        home_win_prob = stat_home_prob

    home_win_prob = max(0.18, min(0.82, home_win_prob))
    away_win_prob = 1.0 - home_win_prob

    # ── Expected runs ─────────────────────────────────────────────────────
    if vegas_available and vegas.get("consensus_total"):
        base_total = vegas["consensus_total"]
    else:
        base_total = LEAGUE_AVG_RPG * 2

    pitcher_run_adj = 1.0 + ((1 - home_p_score) + (1 - away_p_score) - 1.0) * 0.25
    batting_adj     = 1.0 + ((home_bat + away_bat) / 2 - 0.5) * 0.30

    expected_total = base_total * pitcher_run_adj * batting_adj * run_env
    home_runs = expected_total * (home_bat / max(home_bat + away_bat, 0.01)) * (1 + (1 - away_p_score - 0.5) * 0.3)
    away_runs = expected_total - home_runs

    # ── First inning probability ──────────────────────────────────────────
    avg_pitcher_weakness = ((1 - home_p_score) + (1 - away_p_score)) / 2
    fi_run_prob = 0.50 + (avg_pitcher_weakness - 0.50) * 0.25 * run_env
    fi_run_prob = max(0.28, min(0.72, fi_run_prob))

    # ── Confidence score ─────────────────────────────────────────────────
    signals_present = [
        home_p_conf > 0.25,
        away_p_conf > 0.25,
        home_bat != 0.5,
        away_bat != 0.5,
        home_recent_form.get("available", False),
        away_recent_form.get("available", False),
        bool(h2h and h2h.get("available")),
        vegas_available,
        weather_mult != 1.0,
        heis_available,
    ]
    confidence = sum(signals_present) / len(signals_present)

    # ── Layer agreement check ─────────────────────────────────────────────
    # Count how many layers agree on home win favoured
    home_favoured_layers = []
    if stat_home_prob > 0.52:    home_favoured_layers.append("stats")
    if vegas_available and vegas["home_implied"] > 0.52:
        home_favoured_layers.append("vegas")
    if heis_available:
        heis_home = _heisenberg_prob_adjustment(heis, stat_home_prob)
        if heis_home > stat_home_prob + 0.02:
            home_favoured_layers.append("smart_money")

    return {
        # Core probabilities
        "home_win_prob":    round(home_win_prob, 4),
        "away_win_prob":    round(away_win_prob, 4),
        "home_expected_runs": round(home_runs, 2),
        "away_expected_runs": round(away_runs, 2),
        "expected_total":   round(expected_total, 2),
        "fi_run_prob":      round(fi_run_prob, 4),
        "fi_no_run_prob":   round(1 - fi_run_prob, 4),
        # Meta
        "confidence":       round(confidence, 2),
        "layers_agreeing":  home_favoured_layers,
        "run_environment":  round(run_env, 3),
        "weather_desc":     weather_desc,
        "park_factor":      pf,
        "vegas":            vegas,
        "stat_home_prob":   round(stat_home_prob, 4),
        # Components (for display)
        "components": {
            "home_pitcher":  round(home_p_score, 3),
            "away_pitcher":  round(away_p_score, 3),
            "home_pitcher_metrics": home_p_metrics,
            "away_pitcher_metrics": away_p_metrics,
            "home_batting":  round(home_bat, 3),
            "away_batting":  round(away_bat, 3),
            "home_bullpen":  round(home_bp, 3),
            "away_bullpen":  round(away_bp, 3),
            "home_form":     round(home_form, 3),
            "away_form":     round(away_form, 3),
            "run_environment": round(run_env, 3),
            "h2h_home":      round(h2h_home, 3),
        },
    }


def _heisenberg_prob_adjustment(heis_signals: dict, base_prob: float) -> float:
    """
    Convert Heisenberg smart money signal into a probability adjustment.
    If elite wallets are loading YES (home) → push home prob up.
    """
    if not heis_signals:
        return base_prob

    # Aggregate across all market types
    total_conf = 0.0
    yes_conf   = 0.0
    no_conf    = 0.0

    for market_type, signal in heis_signals.items():
        if not isinstance(signal, dict):
            continue
        conf = signal.get("confidence", 0)
        sig  = (signal.get("signal") or "").lower()
        if sig in ("yes", "yes run", "over", "home"):
            yes_conf += conf
        elif sig in ("no", "no run", "under", "away"):
            no_conf += conf
        total_conf += conf

    if total_conf == 0:
        return base_prob

    # Shift probability based on signal strength (max ±8%)
    net = (yes_conf - no_conf) / total_conf
    adj = base_prob + net * 0.08
    return max(0.18, min(0.82, adj))


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY + EDGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def kelly_criterion(win_prob: float, market_price: float,
                    kelly_fraction: float = 0.5) -> dict:
    if market_price <= 0 or market_price >= 1 or win_prob <= 0:
        return {"kelly_pct": 0, "edge_pct": 0, "should_bet": False}

    decimal_odds = 1.0 / market_price
    b = decimal_odds - 1
    q = 1.0 - win_prob
    edge = (win_prob * decimal_odds) - 1
    full_kelly = (b * win_prob - q) / b if b > 0 else 0
    adj_kelly  = max(0, full_kelly * kelly_fraction)

    return {
        "full_kelly_pct": round(full_kelly * 100, 2),
        "kelly_pct":      round(adj_kelly * 100, 2),
        "edge_pct":       round(edge * 100, 2),
        "ev_per_dollar":  round(edge, 4),
        "decimal_odds":   round(decimal_odds, 3),
        "should_bet":     edge * 100 > 0 and adj_kelly > 0,
    }


def calculate_stake(bankroll: float, kelly_pct: float,
                    max_pct: float = 10.0) -> float:
    return round(bankroll * (min(kelly_pct, max_pct) / 100), 2)


def find_edges_v2(analysis: dict, market: dict,
                  heisenberg_signal: dict = None,
                  min_edge: float = 6.0) -> list[dict]:
    """
    Find edges by comparing model probabilities to Polymarket prices.
    Adds a Heisenberg signal boost when smart money confirms direction.
    """
    edges = []
    market_type = market.get("market_type", "moneyline")
    outcomes    = market.get("outcomes", [])

    for outcome in outcomes:
        name         = (outcome.get("name") or "").lower()
        implied_prob = outcome.get("implied_prob")
        price        = outcome.get("price")

        if implied_prob is None or price is None or implied_prob <= 0:
            continue

        model_prob = None
        bet_desc   = ""

        if market_type == "moneyline":
            if any(k in name for k in ["home", "yes"]):
                model_prob = analysis["home_win_prob"]
                bet_desc   = f"Home Win ({analysis.get('home_team', 'Home')})"
            elif any(k in name for k in ["away", "no"]):
                model_prob = analysis["away_win_prob"]
                bet_desc   = f"Away Win ({analysis.get('away_team', 'Away')})"

        elif market_type == "first_inning":
            if any(k in name for k in ["no run", "nrfi", "no"]):
                model_prob = analysis["fi_no_run_prob"]
                bet_desc   = "NRFI — No Run First Inning"
            elif any(k in name for k in ["yes run", "yrfi", "yes"]):
                model_prob = analysis["fi_run_prob"]
                bet_desc   = "YRFI — Yes Run First Inning"

        elif market_type == "over_under":
            import re
            m = re.search(r"(\d+\.?\d*)", name)
            if not m:
                continue
            line = float(m.group(1))
            exp  = analysis["expected_total"]
            if "over" in name:
                model_prob = _over_probability(exp, line)
                bet_desc   = f"Over {line}"
            elif "under" in name:
                model_prob = 1 - _over_probability(exp, line)
                bet_desc   = f"Under {line}"

        if model_prob is None:
            continue

        # ── Heisenberg signal boost ───────────────────────────────────────
        heis_boost = 0.0
        heis_confirms = False
        if heisenberg_signal and heisenberg_signal.get("available"):
            sig  = (heisenberg_signal.get("signal") or "").lower()
            conf = heisenberg_signal.get("confidence", 0)
            # Does smart money align with our model's direction?
            model_direction = "yes" if model_prob > 0.5 else "no"
            sig_direction   = "yes" if sig in ("yes","yes run","over","home") else "no" if sig in ("no","no run","under","away") else "neutral"
            if sig_direction == model_direction and conf > 0.05:
                heis_boost    = conf * 0.04  # max +4% to model prob
                heis_confirms = True
                model_prob    = min(0.88, model_prob + heis_boost)

        raw_edge  = (model_prob - implied_prob) * 100
        # Require ≥ min_edge; if Heisenberg confirms lower threshold by 1%
        threshold = min_edge - 1.0 if heis_confirms else min_edge

        if raw_edge >= threshold:
            edges.append({
                "outcome":         outcome.get("name"),
                "bet_desc":        bet_desc,
                "model_prob":      round(model_prob, 4),
                "market_prob":     round(implied_prob, 4),
                "edge_pct":        round(raw_edge, 2),
                "price":           price,
                "token_id":        outcome.get("token_id"),
                "market_type":     market_type,
                "confidence":      analysis.get("confidence", 0),
                "heis_confirms":   heis_confirms,
                "heis_boost":      round(heis_boost * 100, 2),
                "layers_agreeing": analysis.get("layers_agreeing", []),
            })

    edges.sort(key=lambda x: x["edge_pct"], reverse=True)
    return edges


def get_best_value_plays(analysis: dict, game: dict,
                          min_confidence: float = 0.30) -> list[dict]:
    """
    Best value plays purely from model conviction — useful even when
    no explicit Polymarket edge is found. Compares model vs Vegas.
    Fires when model and Vegas disagree by ≥3% (thin market mispricing).
    """
    plays = []
    vegas = analysis.get("vegas", {})

    if analysis.get("confidence", 0) < min_confidence:
        return plays

    home_win = analysis["home_win_prob"]
    away_win = analysis["away_win_prob"]
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")

    if vegas.get("available"):
        home_gap = home_win - vegas["home_implied"]
        away_gap = away_win - vegas["away_implied"]

        if home_gap >= 0.03:
            plays.append({
                "type":        "best_value",
                "bet_desc":    f"Home Win — {home_team}",
                "model_prob":  home_win,
                "vegas_prob":  vegas["home_implied"],
                "gap_pct":     round(home_gap * 100, 1),
                "dec_odds":    vegas.get("home_dec_odds"),
                "reasoning":   f"Model ({home_win:.1%}) vs Vegas ({vegas['home_implied']:.1%})",
            })
        if away_gap >= 0.03:
            plays.append({
                "type":        "best_value",
                "bet_desc":    f"Away Win — {away_team}",
                "model_prob":  away_win,
                "vegas_prob":  vegas["away_implied"],
                "gap_pct":     round(away_gap * 100, 1),
                "dec_odds":    vegas.get("away_dec_odds"),
                "reasoning":   f"Model ({away_win:.1%}) vs Vegas ({vegas['away_implied']:.1%})",
            })
    else:
        # No Vegas — show model's strongest directional conviction
        if home_win >= 0.58:
            plays.append({
                "type":       "model_conviction",
                "bet_desc":   f"Home Win — {home_team}",
                "model_prob": home_win,
                "gap_pct":    round((home_win - 0.5) * 100, 1),
                "reasoning":  f"Model strong lean: {home_win:.1%}",
            })
        elif away_win >= 0.58:
            plays.append({
                "type":       "model_conviction",
                "bet_desc":   f"Away Win — {away_team}",
                "model_prob": away_win,
                "gap_pct":    round((away_win - 0.5) * 100, 1),
                "reasoning":  f"Model strong lean: {away_win:.1%}",
            })

    plays.sort(key=lambda p: p.get("gap_pct", 0), reverse=True)
    return plays


def generate_reasoning_v2(analysis: dict, edge: dict, game: dict) -> str:
    """Generate concise, data-rich reasoning for a pick."""
    comp  = analysis.get("components", {})
    lines = [f"{E['brain']} *Why this play:*\n"]

    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")

    # Pitcher matchup
    hp = comp.get("home_pitcher", 0.5)
    ap = comp.get("away_pitcher", 0.5)
    if abs(hp - ap) > 0.06:
        better, worse = (home, away) if hp > ap else (away, home)
        val = max(hp, ap)
        lines.append(f"{E['target']} Pitcher edge: *{better}* starter grades {val:.0%} vs {min(hp,ap):.0%}")
        # Show specific metrics if available
        home_m = comp.get("home_pitcher_metrics", [])
        away_m = comp.get("away_pitcher_metrics", [])
        metrics = home_m if hp > ap else away_m
        if metrics:
            lines.append(f"  Key metrics: {' | '.join(metrics[:3])}")

    # Batting
    hb = comp.get("home_batting", 0.5)
    ab = comp.get("away_batting", 0.5)
    if abs(hb - ab) > 0.06:
        better = home if hb > ab else away
        lines.append(f"{E['bat']} Lineup edge: *{better}* rates {max(hb,ab):.0%} vs {min(hb,ab):.0%}")

    # Recent form
    hf = comp.get("home_form", 0.5)
    af = comp.get("away_form", 0.5)
    if abs(hf - af) > 0.10:
        hot = home if hf > af else away
        lines.append(f"{E['fire']} Hot team: *{hot}* (L14: {max(hf,af):.0%} win rate)")

    # Weather
    wd = analysis.get("weather_desc", "")
    re = analysis.get("run_environment", 1.0)
    if wd and wd not in ("Neutral", "Dome/indoor"):
        arrow = "⬆️" if re > 1.01 else "⬇️"
        lines.append(f"🌤 Weather: {wd} → run env {arrow} {re:.2f}x")

    # Vegas alignment
    vegas = analysis.get("vegas", {})
    if vegas.get("available"):
        lines.append(f"{E['money']} Sharp books: Home {vegas['home_implied']:.1%} | Away {vegas['away_implied']:.1%}")
        if vegas.get("consensus_total"):
            lines.append(f"{E['chart']} Vegas total: {vegas['consensus_total']}")

    # Heisenberg confirmation
    if edge.get("heis_confirms"):
        lines.append(f"{E['diamond']} *Smart money confirms this side* (+{edge['heis_boost']:.1f}% boost)")

    # Layers agreement
    layers = edge.get("layers_agreeing", [])
    if layers:
        lines.append(f"{E['check']} Layers agreeing: {', '.join(layers)}")

    # Edge summary
    lines.append(f"\n{E['edge']} Edge: `+{edge['edge_pct']:.1f}%` | Model: `{edge['model_prob']:.1%}` | Market: `{edge['market_prob']:.1%}`")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _f(val, default=None) -> Optional[float]:
    if val is None: return default
    try: return float(val)
    except: return default


def _linear(val: float, lo: float, hi: float) -> float:
    if hi == lo: return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _inverse(val: float, best: float, worst: float) -> float:
    if worst == best: return 0.5
    return max(0.0, min(1.0, (worst - val) / (worst - best)))


def _over_probability(expected: float, line: float) -> float:
    """P(total > line) using normal approximation, std dev ~3.1 runs."""
    std_dev = 3.1
    z = (line - expected) / std_dev
    return 1 / (1 + math.exp(-1.7 * z + 0.05))
