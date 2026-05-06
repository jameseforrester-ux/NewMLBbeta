"""
Alert system — daily digests, live game monitoring, line movement tracking.
Runs as scheduled jobs via APScheduler (built into python-telegram-bot).
"""

import asyncio
import logging
from datetime import datetime

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
    Scheduled job (runs every 2 min during game hours):
    monitors live games for in-game edge opportunities.
    """
    chat_id = context.job.data.get("chat_id")
    if not chat_id:
        return

    try:
        games = await get_todays_schedule()
        live_games = [g for g in games if "In Progress" in g.get("status", "")]

        if not live_games:
            return  # No live games right now

        markets = await get_todays_mlb_markets()
        min_edge = await get_min_edge()

        for game in live_games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            try:
                live = await get_live_game(game_id)
                if not live:
                    continue

                # Track line movements on matching markets
                for market in markets:
                    q = market.get("question", "").lower()
                    away = game.get("away_team", "").lower()
                    home = game.get("home_team", "").lower()

                    if any(team_word in q for team_word in
                           [away.split()[-1], home.split()[-1]]):
                        odds = await get_market_implied_odds(market)
                        await track_line_movement(market["id"], odds.get("outcomes", []))

                        # Check for significant line movement
                        for outcome in odds.get("outcomes", []):
                            name = outcome.get("name", "")
                            mv = get_line_movement(market["id"], name)
                            if mv.get("data_points", 0) >= 3:
                                change = abs(mv.get("change_last", 0))
                                if change >= 3.0:  # 3%+ move
                                    msg = format_line_movement(market, mv)
                                    await context.bot.send_message(
                                        chat_id=chat_id,
                                        text=msg,
                                        parse_mode="Markdown",
                                    )
                                    await log_alert(
                                        "line_move", str(game_id),
                                        f"{name}: {mv['change_last']:+.1f}%"
                                    )

                # Check for in-game edge opportunities
                # (e.g., live total line vs expected remaining runs)
                inning = live.get("inning", 1)
                total_runs = live.get("total_runs", 0)

                if inning >= 3:  # enough game data
                    # Simple in-game expected total
                    runs_per_inning = total_runs / max(inning, 1)
                    remaining_innings = max(9 - inning, 0)
                    projected_total = total_runs + (runs_per_inning * remaining_innings)

                    for market in markets:
                        q = market.get("question", "").lower()
                        if "over" in q or "under" in q or "total" in q:
                            away = game.get("away_team", "").lower()
                            home = game.get("home_team", "").lower()
                            if any(w in q for w in [away.split()[-1], home.split()[-1]]):
                                # Found a total market — check for live edge
                                pass  # would require more complex live modeling

            except Exception as e:
                log.warning(f"Live monitor error for game {game_id}: {e}")

    except Exception as e:
        log.error(f"Live monitor error: {e}")


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
