"""
Edge detection model — calculates true probabilities and compares to market.
Uses a composite weighted model with Log5, park factors, and recent form.
"""

import math
import logging
from typing import Optional
from config import MODEL_WEIGHTS, PARK_FACTORS, E

log = logging.getLogger("model")


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════

def kelly_criterion(win_prob: float, market_price: float,
                    kelly_fraction: float = 0.5) -> dict:
    """
    Calculate Kelly Criterion bet size.

    Args:
        win_prob:        Model's estimated probability of winning (0-1)
        market_price:    Polymarket price / implied probability (0-1)
        kelly_fraction:  Fraction of full Kelly to use (0.5 = half-Kelly)

    Returns:
        dict with kelly_pct, recommended_stake, edge, etc.
    """
    if market_price <= 0 or market_price >= 1 or win_prob <= 0:
        return {"kelly_pct": 0, "edge_pct": 0, "should_bet": False}

    # Decimal odds from market price
    decimal_odds = 1.0 / market_price
    b = decimal_odds - 1  # net profit per unit staked

    # Edge = (probability × odds) - 1
    edge = (win_prob * decimal_odds) - 1
    edge_pct = edge * 100

    # Full Kelly fraction: f* = (bp - q) / b
    q = 1.0 - win_prob
    full_kelly = (b * win_prob - q) / b if b > 0 else 0

    # Apply fractional Kelly
    adjusted_kelly = max(0, full_kelly * kelly_fraction)

    return {
        "full_kelly_pct":  round(full_kelly * 100, 2),
        "kelly_pct":       round(adjusted_kelly * 100, 2),
        "edge_pct":        round(edge_pct, 2),
        "ev_per_dollar":   round(edge, 4),
        "decimal_odds":    round(decimal_odds, 3),
        "should_bet":      edge_pct > 0 and adjusted_kelly > 0,
    }


def calculate_stake(bankroll: float, kelly_pct: float,
                    max_pct: float = 10.0) -> float:
    """Calculate dollar stake from Kelly percentage, with a max cap."""
    pct = min(kelly_pct, max_pct)  # never bet more than max_pct of bankroll
    return round(bankroll * (pct / 100), 2)


# ═══════════════════════════════════════════════════════════════════════════
#  PROBABILITY MODEL
# ═══════════════════════════════════════════════════════════════════════════

def log5_probability(team_a_pct: float, team_b_pct: float) -> float:
    """
    Log5 method — Bill James' formula for matchup probability.
    Given two teams' win percentages, estimates prob of A beating B.

    P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
    """
    pa, pb = team_a_pct, team_b_pct
    if pa + pb == 0 or (pa + pb - 2 * pa * pb) == 0:
        return 0.5
    return (pa - pa * pb) / (pa + pb - 2 * pa * pb)


def pitcher_quality_score(stats: dict) -> float:
    """
    Score a pitcher's quality from 0 (worst) to 1 (best).
    Uses ERA, FIP, WHIP, K/9, BB/9.
    """
    if not stats:
        return 0.5  # league average default

    scores = []

    # ERA (lower is better) — league avg ~4.00
    era = _safe_float(stats.get("era", stats.get("earnedRunAverage")))
    if era is not None:
        scores.append(("pitcher_era", _inverse_score(era, 1.5, 6.0)))

    # FIP (lower is better) — league avg ~4.00
    fip = _safe_float(stats.get("fip"))
    if fip is not None:
        scores.append(("pitcher_fip", _inverse_score(fip, 2.0, 6.0)))

    # WHIP (lower is better) — league avg ~1.30
    whip = _safe_float(stats.get("whip"))
    if whip is not None:
        scores.append(("pitcher_whip", _inverse_score(whip, 0.80, 1.80)))

    # K/9 (higher is better) — league avg ~8.5
    k9 = _safe_float(stats.get("strikeoutsPer9Inn", stats.get("k_per_9")))
    if k9 is not None:
        scores.append(("pitcher_k9", _linear_score(k9, 4.0, 13.0)))

    # BB/9 (lower is better) — league avg ~3.3
    bb9 = _safe_float(stats.get("walksPer9Inn", stats.get("bb_per_9")))
    if bb9 is not None:
        scores.append(("pitcher_bb9", _inverse_score(bb9, 1.0, 6.0)))

    if not scores:
        return 0.5

    # Weighted average using model weights
    total_weight = sum(MODEL_WEIGHTS.get(key, 0.1) for key, _ in scores)
    weighted_sum = sum(MODEL_WEIGHTS.get(key, 0.1) * score for key, score in scores)
    return weighted_sum / max(total_weight, 0.01)


def team_batting_score(stats: dict) -> float:
    """
    Score a team's batting from 0 (worst) to 1 (best).
    Uses OPS, wOBA, wRC+.
    """
    if not stats:
        return 0.5

    scores = []

    # OPS (higher is better) — league avg ~.720
    ops = _safe_float(stats.get("ops"))
    if ops is not None:
        scores.append(("team_ops", _linear_score(ops, 0.600, 0.850)))

    # wOBA (higher is better) — league avg ~.310
    # Note: MLB API might not have wOBA directly, approximate from OBP + SLG
    obp = _safe_float(stats.get("obp"))
    slg = _safe_float(stats.get("slg"))
    if obp and slg:
        woba_approx = obp * 0.69 + slg * 0.31  # rough approximation
        scores.append(("team_woba", _linear_score(woba_approx, 0.200, 0.400)))

    # Runs per game (higher is better)
    runs = _safe_float(stats.get("runs"))
    games = _safe_float(stats.get("gamesPlayed"))
    if runs and games and games > 0:
        rpg = runs / games
        scores.append(("team_wrc_plus", _linear_score(rpg, 3.0, 6.0)))

    if not scores:
        return 0.5

    total_weight = sum(MODEL_WEIGHTS.get(key, 0.1) for key, _ in scores)
    weighted_sum = sum(MODEL_WEIGHTS.get(key, 0.1) * score for key, score in scores)
    return weighted_sum / max(total_weight, 0.01)


def bullpen_score(stats: dict) -> float:
    """Score bullpen quality. Uses team pitching ERA as proxy."""
    if not stats:
        return 0.5
    era = _safe_float(stats.get("era", stats.get("earnedRunAverage")))
    if era is not None:
        return _inverse_score(era, 2.5, 5.5)
    return 0.5


def home_advantage_factor(is_home: bool, team_form: dict = None) -> float:
    """
    Home field advantage modifier.
    MLB home teams win ~54% historically.
    Adjusts based on recent home/away splits if available.
    """
    base = 0.54 if is_home else 0.46

    if team_form and team_form.get("available"):
        results = team_form.get("results", [])
        home_games = [r for r in results if r.get("is_home") == is_home]
        if len(home_games) >= 5:
            wins = sum(1 for g in home_games if g.get("won"))
            split_pct = wins / len(home_games)
            # Blend historical with recent (70/30)
            base = base * 0.7 + split_pct * 0.3

    return base


def park_factor(venue: str) -> float:
    """Get park run factor. >1 = hitter friendly, <1 = pitcher friendly."""
    return PARK_FACTORS.get(venue, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSITE GAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_game(
    home_pitcher_stats: dict,
    away_pitcher_stats: dict,
    home_team_batting: dict,
    away_team_batting: dict,
    home_team_pitching: dict,
    away_team_pitching: dict,
    home_recent_form: dict,
    away_recent_form: dict,
    venue: str,
    h2h: dict = None,
) -> dict:
    """
    Full composite analysis of a game.
    Returns win probabilities, expected runs, edge opportunities.
    """

    # ── Component scores ──────────────────────────────────────────────
    home_p_score = pitcher_quality_score(home_pitcher_stats)
    away_p_score = pitcher_quality_score(away_pitcher_stats)
    home_bat_score = team_batting_score(home_team_batting)
    away_bat_score = team_batting_score(away_team_batting)
    home_bp_score = bullpen_score(home_team_pitching)
    away_bp_score = bullpen_score(away_team_pitching)
    pf = park_factor(venue)

    # ── Recent form ───────────────────────────────────────────────────
    home_form_pct = 0.5
    away_form_pct = 0.5
    if home_recent_form.get("available"):
        home_form_pct = home_recent_form.get("win_pct", 0.5)
    if away_recent_form.get("available"):
        away_form_pct = away_recent_form.get("win_pct", 0.5)

    # ── Composite team strength ───────────────────────────────────────
    w = MODEL_WEIGHTS

    home_strength = (
        (1 - away_p_score) * (w["pitcher_era"] + w["pitcher_fip"]) +  # opponent pitcher weakness
        home_bat_score * (w["team_woba"] + w["team_ops"] + w["team_wrc_plus"]) +
        home_bp_score * w["bullpen_era"] +
        home_advantage_factor(True, home_recent_form) * w["home_away_split"] +
        home_form_pct * w["recent_form"] +
        (1 if pf > 1 else 0.5) * w["park_factor"]
    )

    away_strength = (
        (1 - home_p_score) * (w["pitcher_era"] + w["pitcher_fip"]) +
        away_bat_score * (w["team_woba"] + w["team_ops"] + w["team_wrc_plus"]) +
        away_bp_score * w["bullpen_era"] +
        home_advantage_factor(False, away_recent_form) * w["home_away_split"] +
        away_form_pct * w["recent_form"] +
        (0.5) * w["park_factor"]
    )

    # Normalize to probabilities
    total = home_strength + away_strength
    if total == 0:
        home_win_prob = 0.5
    else:
        home_win_prob = home_strength / total

    # Apply Log5 correction using recent form
    log5_prob = log5_probability(
        max(0.3, min(0.7, home_form_pct)),
        max(0.3, min(0.7, away_form_pct))
    )

    # Blend composite model with Log5 (60/40)
    home_win_prob = home_win_prob * 0.6 + log5_prob * 0.4

    # Clamp to reasonable range
    home_win_prob = max(0.20, min(0.80, home_win_prob))
    away_win_prob = 1 - home_win_prob

    # ── Expected runs ─────────────────────────────────────────────────
    league_avg_rpg = 4.5  # ~2024 league average

    home_expected_runs = (
        league_avg_rpg
        * (1 + (home_bat_score - 0.5) * 0.6)   # batting strength
        * (1 + (1 - away_p_score - 0.5) * 0.5)  # vs opposing pitcher
        * pf                                      # park factor
    )

    away_expected_runs = (
        league_avg_rpg
        * (1 + (away_bat_score - 0.5) * 0.6)
        * (1 + (1 - home_p_score - 0.5) * 0.5)
        * pf
    )

    expected_total = home_expected_runs + away_expected_runs

    # ── First inning scoring probability ──────────────────────────────
    # Avg ~50% of MLB games have a run in the first inning
    base_fi_prob = 0.50

    # Adjust based on starting pitcher quality
    fi_adjustment = ((1 - home_p_score) + (1 - away_p_score)) / 2 - 0.5
    fi_run_prob = base_fi_prob + fi_adjustment * 0.15
    fi_run_prob = max(0.30, min(0.70, fi_run_prob))

    # ── Head to head adjustment ───────────────────────────────────────
    if h2h and h2h.get("available") and h2h.get("games", 0) >= 3:
        h2h_games = h2h["games"]
        # Assuming team1 = home team
        h2h_pct = h2h.get("team1_wins", 0) / h2h_games
        # Small adjustment (10% weight to h2h)
        home_win_prob = home_win_prob * 0.9 + h2h_pct * 0.1
        away_win_prob = 1 - home_win_prob

    # ── Confidence level ──────────────────────────────────────────────
    data_points = sum([
        1 if home_p_score != 0.5 else 0,
        1 if away_p_score != 0.5 else 0,
        1 if home_bat_score != 0.5 else 0,
        1 if away_bat_score != 0.5 else 0,
        1 if home_recent_form.get("available") else 0,
        1 if away_recent_form.get("available") else 0,
        1 if h2h and h2h.get("available") else 0,
    ])
    confidence = min(1.0, data_points / 7.0)

    return {
        "home_win_prob":       round(home_win_prob, 4),
        "away_win_prob":       round(away_win_prob, 4),
        "home_expected_runs":  round(home_expected_runs, 2),
        "away_expected_runs":  round(away_expected_runs, 2),
        "expected_total":      round(expected_total, 2),
        "fi_run_prob":         round(fi_run_prob, 4),
        "fi_no_run_prob":      round(1 - fi_run_prob, 4),
        "park_factor":         pf,
        "confidence":          round(confidence, 2),
        "components": {
            "home_pitcher":  round(home_p_score, 3),
            "away_pitcher":  round(away_p_score, 3),
            "home_batting":  round(home_bat_score, 3),
            "away_batting":  round(away_bat_score, 3),
            "home_bullpen":  round(home_bp_score, 3),
            "away_bullpen":  round(away_bp_score, 3),
            "home_form":     round(home_form_pct, 3),
            "away_form":     round(away_form_pct, 3),
            "log5":          round(log5_prob, 4),
        },
    }


def find_edges(analysis: dict, market_odds: dict, min_edge: float = 6.0) -> list[dict]:
    """
    Compare model probabilities vs market implied odds.
    Returns list of edges found.
    """
    edges = []

    outcomes = market_odds.get("outcomes", [])
    market_type = market_odds.get("market_type", "moneyline")

    for outcome in outcomes:
        name = outcome.get("name", "").lower()
        implied_prob = outcome.get("implied_prob")
        price = outcome.get("price")

        if implied_prob is None or price is None:
            continue

        model_prob = None
        bet_description = ""

        # Match outcome to model probability
        if market_type == "moneyline":
            if any(kw in name for kw in ["home", "yes"]):
                model_prob = analysis.get("home_win_prob")
                bet_description = "Home Win"
            elif any(kw in name for kw in ["away", "no"]):
                model_prob = analysis.get("away_win_prob")
                bet_description = "Away Win"

        elif market_type == "first_inning":
            if any(kw in name for kw in ["no run", "nrfi", "scoreless", "no"]):
                model_prob = analysis.get("fi_no_run_prob")
                bet_description = "NRFI (No Run First Inning)"
            elif any(kw in name for kw in ["yes", "yrfi", "run"]):
                model_prob = analysis.get("fi_run_prob")
                bet_description = "YRFI (Yes Run First Inning)"

        elif market_type == "over_under":
            # For O/U, check if the expected total suggests value
            expected = analysis.get("expected_total", 0)
            if "over" in name:
                # Extract the line number from the outcome name
                line = _extract_number(name)
                if line and expected > line:
                    model_prob = _total_over_prob(expected, line)
                    bet_description = f"Over {line}"
            elif "under" in name:
                line = _extract_number(name)
                if line and expected < line:
                    model_prob = 1 - _total_over_prob(expected, line)
                    bet_description = f"Under {line}"

        if model_prob is not None:
            edge_pct = (model_prob - implied_prob) * 100

            if edge_pct >= min_edge:
                edges.append({
                    "outcome":      outcome.get("name"),
                    "bet_desc":     bet_description,
                    "model_prob":   round(model_prob, 4),
                    "market_prob":  round(implied_prob, 4),
                    "edge_pct":     round(edge_pct, 2),
                    "price":        price,
                    "token_id":     outcome.get("token_id"),
                    "market_type":  market_type,
                    "confidence":   analysis.get("confidence", 0),
                })

    # Sort by edge descending
    edges.sort(key=lambda x: x["edge_pct"], reverse=True)
    return edges


def generate_reasoning(analysis: dict, edge: dict, game: dict) -> str:
    """Generate human-readable reasoning for a bet recommendation."""
    comp = analysis.get("components", {})
    lines = []

    lines.append(f"{E['brain']} *Model Analysis*")
    lines.append("")

    # Pitcher matchup
    home_p = comp.get("home_pitcher", 0.5)
    away_p = comp.get("away_pitcher", 0.5)
    if home_p > away_p:
        lines.append(f"{E['target']} Pitching edge: Home starter grades higher ({home_p:.0%} vs {away_p:.0%})")
    elif away_p > home_p:
        lines.append(f"{E['target']} Pitching edge: Away starter grades higher ({away_p:.0%} vs {home_p:.0%})")

    # Batting
    home_b = comp.get("home_batting", 0.5)
    away_b = comp.get("away_batting", 0.5)
    if abs(home_b - away_b) > 0.05:
        better = "Home" if home_b > away_b else "Away"
        lines.append(f"{E['bat']} Batting advantage: {better} team ({max(home_b, away_b):.0%} vs {min(home_b, away_b):.0%})")

    # Recent form
    home_f = comp.get("home_form", 0.5)
    away_f = comp.get("away_form", 0.5)
    if abs(home_f - away_f) > 0.1:
        hot = "Home" if home_f > away_f else "Away"
        lines.append(f"{E['fire']} {hot} team is hotter (L14 win%: {max(home_f, away_f):.0%} vs {min(home_f, away_f):.0%})")

    # Park factor
    pf = analysis.get("park_factor", 1.0)
    if pf > 1.03:
        lines.append(f"{E['park']} Hitter-friendly park (factor: {pf:.2f})")
    elif pf < 0.97:
        lines.append(f"{E['park']} Pitcher-friendly park (factor: {pf:.2f})")

    lines.append("")
    lines.append(f"{E['edge']} *Edge Breakdown*")
    lines.append(f"  Model prob: {edge['model_prob']:.1%}")
    lines.append(f"  Market prob: {edge['market_prob']:.1%}")
    lines.append(f"  Edge: +{edge['edge_pct']:.1f}%")
    lines.append(f"  Confidence: {analysis.get('confidence', 0):.0%}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _linear_score(val: float, low: float, high: float) -> float:
    """Map value to 0-1 range where low=0 and high=1."""
    if high == low:
        return 0.5
    return max(0, min(1, (val - low) / (high - low)))


def _inverse_score(val: float, best: float, worst: float) -> float:
    """Map value to 0-1 range where best=1 and worst=0 (lower is better)."""
    if worst == best:
        return 0.5
    return max(0, min(1, (worst - val) / (worst - best)))


def _extract_number(text: str) -> Optional[float]:
    """Extract a number from text like 'Over 8.5'."""
    import re
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def _total_over_prob(expected_total: float, line: float) -> float:
    """
    Estimate probability of going over a total line.
    Uses a simplified normal approximation.
    MLB run totals have std dev of ~3.2 runs.
    """
    std_dev = 3.2
    z = (line - expected_total) / std_dev
    # Approximate CDF using logistic function
    prob_under = 1 / (1 + math.exp(-1.7 * z))
    return 1 - prob_under
