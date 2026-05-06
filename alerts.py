"""
Alert system — daily digests, live game monitoring, line movement tracking.
Runs as scheduled jobs via APScheduler (built into python-telegram-bot).

Live monitor polls every 5 minutes but only fires alerts when a meaningful
threshold is crossed — no spam, only actionable signals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from config import E
from mlb_data import (
    get_todays_schedule, get_live_game,
    get_team_stats, get_pitcher_detailed,
    get_team_recent_form, get_head_to_head, search_team_id,
)
from polymarket import (
    get_todays_mlb_markets, get_market_implied_odds,
    track_line_movement, get_line_movement,
)
from model import analyze_game, find_edges, kelly_criterion, calculate_stake, generate_reasoning
from database import (
    get_bankroll, get_min_edge, get_kelly_fraction,
    log_alert, log_bet,
)
from formatters import (
    format_daily_digest, format_edge_alert, format_live_alert,
    format_line_movement,
)

log = logging.getLogger("alerts")

# ── Alert thresholds (tune these without touching logic) ─────────────────────
THRESHOLDS = {
    "line_move_pct":        5.0,   # % swing before flagging a line move
    "live_edge_pct":        6.0,   # % edge needed to fire a live bet alert
    "big_inning_runs":      3,     # runs in one inning = big inning alert
    "blowout_lead":         6,     # run lead after 6th = suppress alerts
    "pitcher_pull_inning":  5,     # starter pulled before this inning = alert
    "total_pace_gap":       1.5,   # projected total vs line gap (runs) for O/U alert
    "cooldown_minutes":     20,    # min minutes between same alert type per game
}

# ── Per-game state memory (in-process, resets on restart) ───────────────────
# Tracks previous game state so we only alert on *changes*
_game_state: dict[int, dict] = {}

# Tracks last alert time per (game_id, alert_type) to enforce cooldown
_alert_cooldown: dict[str, datetime] = {}


def _on_cooldown(game_id: int, alert_type: str) -> bool:
    """Return True if this alert type fired too recently for this game."""
    key = f"{game_id}_{alert_type}"
    last = _alert_cooldown.get(key)
    if last is None:
        return False
    elapsed = (datetime.utcnow() - last).total_seconds() / 60
    return elapsed < THRESHOLDS["cooldown_minutes"]


def _mark_fired(game_id: int, alert_type: str):
    """Record that an alert just fired."""
    _alert_cooldown[f"{game_id}_{alert_type}"] = datetime.utcnow()


async def run_daily_digest(context) -> None:
    """
    Scheduled job: scans all today's games, runs the model,
    compares to Polymarket, sends digest of top edges.
    """
    chat_id = context.job.data.get("chat_id")
    if not chat_id:
        return

    log.info("Running daily digest...")

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{E['refresh']} Scanning today's slate...",
        )

        games = await get_todays_schedule()
        markets = await get_todays_mlb_markets()
        bankroll = await get_bankroll()
        min_edge = await get_min_edge()
        kelly_frac = await get_kelly_fraction()

        all_picks = []

        for game in games:
            if game.get("status") not in ("Scheduled", "Pre-Game", "Warmup"):
                continue

            try:
                analysis, edges = await _analyze_single_game(game, markets, min_edge)
                if not edges:
                    continue

                for edge in edges:
                    kelly = kelly_criterion(edge["model_prob"], edge["price"], kelly_frac)
                    if not kelly["should_bet"]:
                        continue

                    stake = calculate_stake(bankroll, kelly["kelly_pct"])
                    all_picks.append({
                        "game":     game,
                        "analysis": analysis,
                        "edge":     edge,
                        "kelly":    kelly,
                        "stake":    stake,
                    })
            except Exception as e:
                log.warning(f"Error analyzing game {game.get('game_id')}: {e}")
                continue

        # Sort by edge descending
        all_picks.sort(key=lambda p: p["edge"]["edge_pct"], reverse=True)

        # Send digest
        digest_msg = format_daily_digest(all_picks)
        await context.bot.send_message(
            chat_id=chat_id,
            text=digest_msg,
            parse_mode="Markdown",
        )

        # Send individual detailed cards for top 3
        for pick in all_picks[:3]:
            reasoning = generate_reasoning(
                pick["analysis"], pick["edge"], pick["game"]
            )
            alert_msg = format_edge_alert(
                pick["game"], pick["edge"],
                pick["analysis"], pick["kelly"], pick["stake"]
            )
            full_msg = f"{alert_msg}\n\n{reasoning}"

            await context.bot.send_message(
                chat_id=chat_id,
                text=full_msg,
                parse_mode="Markdown",
            )
            await asyncio.sleep(1)

        await log_alert("digest", "", f"Sent {len(all_picks)} picks")

    except Exception as e:
        log.error(f"Daily digest error: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{E['x']} Digest error: {str(e)[:200]}",
        )


async def run_live_monitor(context) -> None:
    """
    Polls every 5 minutes but only sends alerts when a threshold is crossed.

    Triggers that fire an alert:
      1. Line movement ≥5% on any matching Polymarket market
      2. Starting pitcher pulled before inning 5
      3. Big inning (≥3 runs in one half-inning)
      4. Live O/U pace gap ≥1.5 runs vs the market line (inning 3+)
      5. Score swing creates new moneyline edge ≥6% vs market
      6. First inning resolved with a NRFI/YRFI position recommended

    Suppressed when:
      - No live games
      - Game is a blowout (6+ run lead after 6th inning)
      - Same alert already fired within cooldown window (20 min)
    """
    chat_id = context.job.data.get("chat_id")
    if not chat_id:
        return

    try:
        games = await get_todays_schedule()
        live_games = [g for g in games if "In Progress" in g.get("status", "")]
        if not live_games:
            return

        markets = await get_todays_mlb_markets()
        min_edge = await get_min_edge()
        bankroll = await get_bankroll()
        kelly_frac = await get_kelly_fraction()

        for game in live_games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            try:
                live = await get_live_game(game_id)
                if not live:
                    continue

                inning      = live.get("inning", 1) or 1
                inning_half = live.get("inning_half", "Top")
                away_score  = live.get("away_score", 0) or 0
                home_score  = live.get("home_score", 0) or 0
                total_runs  = away_score + home_score
                run_diff    = abs(away_score - home_score)

                prev = _game_state.get(game_id, {})

                # ── Blowout filter ────────────────────────────────────────
                if inning >= 6 and run_diff >= THRESHOLDS["blowout_lead"]:
                    _game_state[game_id] = _snapshot(live)
                    continue  # no useful edges in a blowout

                away_kw = game.get("away_team", "").lower().split()[-1]
                home_kw = game.get("home_team", "").lower().split()[-1]
                away_abbr = game.get("away_abbrev", "???")
                home_abbr = game.get("home_abbrev", "???")

                # ── 1. Line movement alert ────────────────────────────────
                for market in markets:
                    q = market.get("question", "").lower()
                    if not any(kw in q for kw in [away_kw, home_kw]):
                        continue

                    odds = await get_market_implied_odds(market)
                    await track_line_movement(market["id"], odds.get("outcomes", []))

                    for outcome in odds.get("outcomes", []):
                        oname = outcome.get("name", "")
                        mv = get_line_movement(market["id"], oname)
                        if mv.get("data_points", 0) < 3:
                            continue

                        change = abs(mv.get("change_last", 0))
                        if change >= THRESHOLDS["line_move_pct"]:
                            direction = "▲" if mv["direction"] == "up" else "▼"
                            alert_key = f"line_{market['id']}_{oname}"
                            if not _on_cooldown(game_id, alert_key):
                                msg = (
                                    f"{E['chart']} *Line Movement Alert*\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"{E['vs']} `{away_abbr}` vs `{home_abbr}` — Inning {inning}\n\n"
                                    f"{E['mag']} {market.get('question', '')[:60]}\n\n"
                                    f"  Outcome:  *{oname}*\n"
                                    f"  Previous: `{mv.get('previous', 0):.1%}`\n"
                                    f"  Current:  `{mv.get('current', 0):.1%}` {direction}\n"
                                    f"  Move:     `{mv.get('change_last', 0):+.1f}%`\n"
                                    f"  Since open: `{mv.get('change_open', 0):+.1f}%`\n\n"
                                    f"{E['brain']} Sharp money may be moving — check for edge."
                                )
                                await context.bot.send_message(
                                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                                )
                                _mark_fired(game_id, alert_key)
                                await log_alert("line_move", str(game_id),
                                                f"{oname}: {mv['change_last']:+.1f}%")

                # ── 2. Pitcher pulled early ───────────────────────────────
                prev_inning = prev.get("inning", 0)
                if inning >= THRESHOLDS["pitcher_pull_inning"] and prev_inning < THRESHOLDS["pitcher_pull_inning"]:
                    # Inning just crossed threshold — check if starter still in
                    plays = live.get("plays", {})
                    current_play = plays.get("currentPlay", {})
                    matchup = current_play.get("matchup", {})
                    pitcher_name = matchup.get("pitcher", {}).get("fullName", "")

                    orig_home_p = game.get("home_pitcher", {}).get("name", "")
                    orig_away_p = game.get("away_pitcher", {}).get("name", "")

                    for side, orig, abbr in [
                        ("home", orig_home_p, home_abbr),
                        ("away", orig_away_p, away_abbr),
                    ]:
                        if (orig and orig != "TBD"
                                and pitcher_name
                                and orig.split()[-1] not in pitcher_name):
                            if not _on_cooldown(game_id, f"pitcher_pull_{side}"):
                                msg = (
                                    f"{E['alert']} *Starter Pulled — Edge Shift*\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"{E['vs']} `{away_abbr}` vs `{home_abbr}`\n\n"
                                    f"{E['target']} {abbr} starter *{orig}* appears out\n"
                                    f"  Inning: {inning} | Score: {away_score}-{home_score}\n\n"
                                    f"{E['brain']} Bullpen usage changes win probability.\n"
                                    f"Check moneyline & run line markets for new edge."
                                )
                                await context.bot.send_message(
                                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                                )
                                _mark_fired(game_id, f"pitcher_pull_{side}")
                                await log_alert("pitcher_pull", str(game_id), f"{abbr}: {orig}")

                # ── 3. Big inning alert ───────────────────────────────────
                prev_away = prev.get("away_score", 0)
                prev_home = prev.get("home_score", 0)
                inning_runs_away = away_score - prev_away
                inning_runs_home = home_score - prev_home
                inning_runs = inning_runs_away + inning_runs_home

                if inning_runs >= THRESHOLDS["big_inning_runs"]:
                    if not _on_cooldown(game_id, "big_inning"):
                        scoring_team = (
                            away_abbr if inning_runs_away >= inning_runs_home else home_abbr
                        )
                        msg = (
                            f"{E['fire']} *Big Inning — Total Alert*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"{E['vs']} `{away_abbr}` {away_score} - {home_score} `{home_abbr}`\n"
                            f"  Inning: {inning} | {inning_half}\n\n"
                            f"{E['run']} *{inning_runs} runs* scored — {scoring_team} exploding\n\n"
                            f"{E['chart']} Live total now: `{total_runs}`\n"
                            f"{E['brain']} Check the over/under market for live value."
                        )
                        await context.bot.send_message(
                            chat_id=chat_id, text=msg, parse_mode="Markdown"
                        )
                        _mark_fired(game_id, "big_inning")
                        await log_alert("big_inning", str(game_id),
                                        f"{inning_runs} runs in inning {inning}")

                # ── 4. Live O/U pace vs market line (inning 3+) ──────────
                if inning >= 3 and not _on_cooldown(game_id, "ou_pace"):
                    runs_per_inning = total_runs / inning
                    projected_final = total_runs + (runs_per_inning * max(9 - inning, 0))

                    for market in markets:
                        q = market.get("question", "").lower()
                        if market.get("market_type") != "over_under":
                            continue
                        if not any(kw in q for kw in [away_kw, home_kw]):
                            continue

                        for outcome in market.get("outcomes", []):
                            oname = (outcome.get("name") or "").lower()
                            implied = outcome.get("implied_prob")
                            if implied is None:
                                continue

                            import re
                            m = re.search(r"(\d+\.?\d*)", oname)
                            if not m:
                                continue
                            line = float(m.group(1))

                            gap = abs(projected_final - line)
                            if gap >= THRESHOLDS["total_pace_gap"]:
                                direction_word = "OVER" if projected_final > line else "UNDER"
                                msg = (
                                    f"{E['bolt']} *Live O/U Pace Alert*\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"{E['vs']} `{away_abbr}` {away_score} - {home_score} `{home_abbr}`\n"
                                    f"  Inning: {inning}\n\n"
                                    f"{E['chart']} Market line:  `{line}`\n"
                                    f"{E['run']} Pace projects: `{projected_final:.1f}` total runs\n"
                                    f"{E['target']} Gap: `{gap:+.1f}` favours *{direction_word}*\n\n"
                                    f"{E['brain']} Live market may be slow to adjust — check now."
                                )
                                await context.bot.send_message(
                                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                                )
                                _mark_fired(game_id, "ou_pace")
                                await log_alert("ou_pace", str(game_id),
                                                f"proj {projected_final:.1f} vs line {line}")
                                break

                # ── 5. Live moneyline edge check (score change) ───────────
                score_changed = (
                    away_score != prev.get("away_score", away_score) or
                    home_score != prev.get("home_score", home_score)
                )
                if score_changed and not _on_cooldown(game_id, "live_edge"):
                    try:
                        analysis, edges = await _analyze_single_game(
                            game, markets, THRESHOLDS["live_edge_pct"]
                        )
                        for edge in edges:
                            if edge.get("market_type") == "moneyline":
                                kelly = kelly_criterion(
                                    edge["model_prob"], edge["price"], kelly_frac
                                )
                                stake = calculate_stake(bankroll, kelly["kelly_pct"])
                                if kelly["should_bet"]:
                                    alert_msg = format_edge_alert(
                                        game, edge, analysis, kelly, stake
                                    )
                                    header = (
                                        f"{E['diamond']} *Live Edge — Score Change*\n"
                                        f"Score is now {away_abbr} {away_score} - "
                                        f"{home_score} {home_abbr}\n\n"
                                    )
                                    await context.bot.send_message(
                                        chat_id=chat_id,
                                        text=header + alert_msg,
                                        parse_mode="Markdown",
                                    )
                                    _mark_fired(game_id, "live_edge")
                                    await log_alert("live_edge", str(game_id),
                                                    f"{edge['bet_desc']} +{edge['edge_pct']:.1f}%")
                                    break
                    except Exception as e:
                        log.warning(f"Live edge check error game {game_id}: {e}")

                # ── 6. NRFI resolved alert ────────────────────────────────
                prev_inn = prev.get("inning", 0)
                if prev_inn == 1 and inning == 2 and not _on_cooldown(game_id, "nrfi_result"):
                    result_word = "✅ NRFI holds" if total_runs == 0 else f"❌ YRFI — {total_runs} run(s) scored"
                    msg = (
                        f"{E['dice']} *First Inning Resolved*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"{E['vs']} `{away_abbr}` vs `{home_abbr}`\n\n"
                        f"{result_word}\n"
                        f"Score after 1st: {away_score} - {home_score}"
                    )
                    await context.bot.send_message(
                        chat_id=chat_id, text=msg, parse_mode="Markdown"
                    )
                    _mark_fired(game_id, "nrfi_result")

                # ── Update game state ─────────────────────────────────────
                _game_state[game_id] = _snapshot(live)

            except Exception as e:
                log.warning(f"Live monitor error for game {game_id}: {e}")

    except Exception as e:
        log.error(f"Live monitor error: {e}")


def _snapshot(live: dict) -> dict:
    """Save a lightweight snapshot of current game state."""
    return {
        "inning":      live.get("inning", 0),
        "inning_half": live.get("inning_half", ""),
        "away_score":  live.get("away_score", 0),
        "home_score":  live.get("home_score", 0),
        "total_runs":  live.get("total_runs", 0),
    }


async def run_injury_check(context) -> None:
    """
    Scheduled job: checks for late-breaking lineup/injury news.
    Uses MLB API roster status changes.
    """
    chat_id = context.job.data.get("chat_id")
    if not chat_id:
        return

    try:
        games = await get_todays_schedule()

        for game in games:
            # Check if probable pitcher changed from earlier scan
            for side in ["away", "home"]:
                pitcher = game.get(f"{side}_pitcher", {})
                if pitcher.get("name") == "TBD":
                    team_name = game.get(f"{side}_team", "")
                    msg = (
                        f"{E['alert']} *Pitcher Update*\n"
                        f"{team_name}: Starting pitcher is TBD\n"
                        f"Game: {game.get('away_abbrev')} @ {game.get('home_abbrev')}\n\n"
                        f"This may create edge opportunities once announced."
                    )
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=msg,
                        parse_mode="Markdown",
                    )

    except Exception as e:
        log.error(f"Injury check error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

async def _analyze_single_game(game: dict, markets: list[dict],
                                min_edge: float) -> tuple[dict, list[dict]]:
    """Run full analysis on a single game and find edges."""

    # Fetch all data concurrently
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    home_pid = game.get("home_pitcher", {}).get("id")
    away_pid = game.get("away_pitcher", {}).get("id")

    tasks = [
        get_team_recent_form(home_team),
        get_team_recent_form(away_team),
        get_head_to_head(home_team, away_team),
    ]

    if home_pid:
        tasks.append(get_pitcher_detailed(home_pid))
    if away_pid:
        tasks.append(get_pitcher_detailed(away_pid))

    # Get team IDs for stats
    home_tid = await search_team_id(home_team)
    away_tid = await search_team_id(away_team)
    if home_tid:
        tasks.append(get_team_stats(home_tid))
    if away_tid:
        tasks.append(get_team_stats(away_tid))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results safely
    idx = 0
    home_form = results[idx] if not isinstance(results[idx], Exception) else {"available": False}
    idx += 1
    away_form = results[idx] if not isinstance(results[idx], Exception) else {"available": False}
    idx += 1
    h2h = results[idx] if not isinstance(results[idx], Exception) else {"available": False}
    idx += 1

    home_pitcher_stats = {}
    if home_pid:
        hp = results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else {}
        home_pitcher_stats = hp.get("season", {})
        idx += 1

    away_pitcher_stats = {}
    if away_pid:
        ap = results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else {}
        away_pitcher_stats = ap.get("season", {})
        idx += 1

    home_team_stats = {}
    if home_tid and idx < len(results):
        hts = results[idx] if not isinstance(results[idx], Exception) else {}
        home_team_stats = hts
        idx += 1

    away_team_stats = {}
    if away_tid and idx < len(results):
        ats = results[idx] if not isinstance(results[idx], Exception) else {}
        away_team_stats = ats
        idx += 1

    # Run model
    analysis = analyze_game(
        home_pitcher_stats=home_pitcher_stats,
        away_pitcher_stats=away_pitcher_stats,
        home_team_batting=home_team_stats.get("hitting", {}),
        away_team_batting=away_team_stats.get("hitting", {}),
        home_team_pitching=home_team_stats.get("pitching", {}),
        away_team_pitching=away_team_stats.get("pitching", {}),
        home_recent_form=home_form,
        away_recent_form=away_form,
        venue=game.get("venue", ""),
        h2h=h2h,
    )

    # Find matching markets and edges
    all_edges = []
    for market in markets:
        q = market.get("question", "").lower()
        away_kw = away_team.lower().split()[-1]
        home_kw = home_team.lower().split()[-1]

        if any(kw in q for kw in [away_kw, home_kw]):
            odds_data = await get_market_implied_odds(market)
            odds_data["market_type"] = market.get("market_type", "moneyline")
            edges = find_edges(analysis, odds_data, min_edge)
            all_edges.extend(edges)

    return analysis, all_edges
