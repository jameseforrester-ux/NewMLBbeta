"""
Rich Telegram message formatters — builds beautiful inline messages
with emojis, tables, and clear visual hierarchy.
"""

from config import E, TEAM_ABBREV
from datetime import datetime


def format_main_menu() -> str:
    return (
        f"{E['base']} *MLB Edge Bot* {E['edge']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Your AI-powered MLB betting edge.\n"
        f"Analyzing Statcast data against\n"
        f"Polymarket lines in real-time.\n\n"
        f"{E['diamond']} *Quick Actions*\n"
        f"Use the menu below to navigate."
    )


def format_game_card(game: dict) -> str:
    """Format a single game as a compact card."""
    status = game.get("status", "")
    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")

    # Status indicator
    if "In Progress" in status or "Live" in status:
        inning = game.get("inning", "?")
        half = "▲" if "Top" in game.get("inning_state", "") else "▼"
        status_str = f"{E['green']} LIVE {half}{inning}"
        score = f"{game.get('away_score', 0)} - {game.get('home_score', 0)}"
    elif "Final" in status:
        status_str = f"{E['check']} Final"
        score = f"{game.get('away_score', 0)} - {game.get('home_score', 0)}"
    elif "Scheduled" in status or "Pre-Game" in status:
        gt = game.get("game_date", "")
        try:
            dt = datetime.fromisoformat(gt.replace("Z", "+00:00"))
            time_str = dt.strftime("%I:%M %p")
        except:
            time_str = "TBD"
        status_str = f"{E['clock']} {time_str}"
        score = "vs"
    else:
        status_str = f"{E['yellow']} {status}"
        score = "—"

    # Pitchers
    away_p = game.get("away_pitcher", {}).get("name", "TBD")
    home_p = game.get("home_pitcher", {}).get("name", "TBD")

    return (
        f"┌─────────────────────┐\n"
        f"│ {status_str}\n"
        f"│ {E['away']} `{away:>3}` {score:^7} `{home:<3}` {E['home']}\n"
        f"│ {E['target']} {away_p}\n"
        f"│ {E['target']} {home_p}\n"
        f"│ {E['park']} {game.get('venue', 'TBD')}\n"
        f"└─────────────────────┘"
    )


def format_games_list(games: list[dict]) -> str:
    """Format today's full game slate."""
    if not games:
        return f"{E['x']} No games found for today."

    header = (
        f"{E['calendar']} *Today's MLB Slate*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{len(games)} games scheduled\n\n"
    )

    cards = []
    for g in games:
        cards.append(format_game_card(g))

    return header + "\n".join(cards)


def format_analysis(game: dict, analysis: dict) -> str:
    """Format full game analysis."""
    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")
    comp = analysis.get("components", {})

    conf_emoji = (
        E["green"] if analysis["confidence"] >= 0.7
        else E["yellow"] if analysis["confidence"] >= 0.4
        else E["red"]
    )

    return (
        f"{E['brain']} *Game Analysis*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{E['vs']} `{away}` vs `{home}`\n"
        f"{E['park']} {game.get('venue', 'TBD')}\n\n"

        f"{E['chart']} *Win Probability*\n"
        f"  {away}: `{analysis['away_win_prob']:.1%}`\n"
        f"  {home}: `{analysis['home_win_prob']:.1%}`\n\n"

        f"{E['run']} *Expected Runs*\n"
        f"  {away}: `{analysis['away_expected_runs']:.1f}`\n"
        f"  {home}: `{analysis['home_expected_runs']:.1f}`\n"
        f"  Total: `{analysis['expected_total']:.1f}`\n\n"

        f"{E['dice']} *First Inning*\n"
        f"  NRFI: `{analysis['fi_no_run_prob']:.1%}`\n"
        f"  YRFI: `{analysis['fi_run_prob']:.1%}`\n\n"

        f"{E['mag']} *Component Scores*\n"
        f"```\n"
        f"{'Metric':<16} {'Away':>6} {'Home':>6}\n"
        f"{'─'*30}\n"
        f"{'Pitcher':<16} {comp.get('away_pitcher', 0.5):>5.0%}  {comp.get('home_pitcher', 0.5):>5.0%}\n"
        f"{'Batting':<16} {comp.get('away_batting', 0.5):>5.0%}  {comp.get('home_batting', 0.5):>5.0%}\n"
        f"{'Bullpen':<16} {comp.get('away_bullpen', 0.5):>5.0%}  {comp.get('home_bullpen', 0.5):>5.0%}\n"
        f"{'Recent Form':<16} {comp.get('away_form', 0.5):>5.0%}  {comp.get('home_form', 0.5):>5.0%}\n"
        f"{'Log5 (home)':<16} {'':>6} {comp.get('log5', 0.5):>5.0%}\n"
        f"```\n\n"
        f"{conf_emoji} Confidence: `{analysis['confidence']:.0%}`\n"
        f"{E['park']} Park Factor: `{analysis['park_factor']:.2f}`"
    )


def format_edge_alert(game: dict, edge: dict, analysis: dict,
                      kelly: dict, stake: float) -> str:
    """Format an edge alert / bet recommendation."""
    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")

    edge_bar = _make_bar(edge["edge_pct"], max_val=20)

    return (
        f"{E['alert']} *EDGE DETECTED* {E['alert']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{E['vs']} `{away}` vs `{home}`\n"
        f"{E['target']} *{edge['bet_desc']}*\n\n"

        f"{E['chart']} *Probability Comparison*\n"
        f"  Model:  `{edge['model_prob']:.1%}`\n"
        f"  Market: `{edge['market_prob']:.1%}`\n"
        f"  Edge:   `+{edge['edge_pct']:.1f}%` {edge_bar}\n\n"

        f"{E['money']} *Position Sizing (Kelly)*\n"
        f"  Full Kelly:  `{kelly['full_kelly_pct']:.1f}%`\n"
        f"  Adj. Kelly:  `{kelly['kelly_pct']:.1f}%`\n"
        f"  Rec. Stake:  `${stake:.2f}`\n"
        f"  EV/Dollar:   `{kelly['ev_per_dollar']:.3f}`\n\n"

        f"{E['edge']} *Market*: {edge.get('market_type', 'ML').upper()}\n"
        f"  Price: `{edge['price']:.3f}`\n"
        f"  Odds:  `{kelly['decimal_odds']:.2f}x`\n\n"

        f"{_confidence_badge(analysis.get('confidence', 0))}"
    )


def format_daily_digest(picks: list[dict]) -> str:
    """Format the daily digest of top picks."""
    if not picks:
        return (
            f"{E['calendar']} *Daily Edge Digest*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{E['mag']} No edges found today meeting\n"
            f"your minimum threshold.\n\n"
            f"Markets may be efficiently priced\n"
            f"or data is still loading.\n"
            f"Check back closer to game time."
        )

    header = (
        f"{E['fire']} *Daily Edge Digest*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{E['calendar']} {datetime.now().strftime('%A, %B %d %Y')}\n"
        f"{E['target']} {len(picks)} edge(s) found\n\n"
    )

    pick_strs = []
    for i, p in enumerate(picks[:10], 1):
        edge_bar = _make_bar(p["edge"]["edge_pct"], max_val=20)
        pick_strs.append(
            f"*{i}. {p['game'].get('away_abbrev', '?')} @ "
            f"{p['game'].get('home_abbrev', '?')}*\n"
            f"  {E['target']} {p['edge']['bet_desc']}\n"
            f"  {E['chart']} Edge: `+{p['edge']['edge_pct']:.1f}%` {edge_bar}\n"
            f"  {E['money']} Stake: `${p['stake']:.2f}` "
            f"(Kelly: {p['kelly']['kelly_pct']:.1f}%)\n"
            f"  {_confidence_badge(p['analysis'].get('confidence', 0))}\n"
        )

    return header + "\n".join(pick_strs)


def format_pnl(summary: dict, bankroll: float) -> str:
    """Format P&L summary."""
    pnl = summary.get("total_pnl", 0)
    pnl_emoji = E["up"] if pnl >= 0 else E["down"]
    roi = (pnl / max(summary.get("total_staked", 1), 1)) * 100

    total = summary.get("total_bets", 0)
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pending = summary.get("pending", 0)
    win_rate = (wins / max(wins + losses, 1)) * 100

    return (
        f"{E['trophy']} *Performance Dashboard*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{E['bank']} Bankroll: `${bankroll:.2f}`\n\n"

        f"{pnl_emoji} *P&L: `${pnl:+.2f}`*\n"
        f"  ROI: `{roi:+.1f}%`\n"
        f"  Total Staked: `${summary.get('total_staked', 0):.2f}`\n\n"

        f"{E['chart']} *Record*\n"
        f"```\n"
        f"{'Total Bets:':<16} {total:>5}\n"
        f"{'Wins:':<16} {wins:>5} {E['green']}\n"
        f"{'Losses:':<16} {losses:>5} {E['red']}\n"
        f"{'Pending:':<16} {pending:>5} {E['yellow']}\n"
        f"{'Win Rate:':<16} {win_rate:>4.1f}%\n"
        f"{'Avg Edge:':<16} {summary.get('avg_edge', 0):>4.1f}%\n"
        f"```"
    )


def format_settings(bankroll: float, min_edge: float,
                    kelly_frac: float) -> str:
    """Format current settings display."""
    return (
        f"{E['gear']} *Bot Settings*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{E['bank']}  Bankroll:     `${bankroll:.2f}`\n"
        f"{E['edge']}  Min Edge:     `{min_edge:.1f}%`\n"
        f"{E['dice']}  Kelly Frac:   `{kelly_frac:.0%}`\n"
        f"{E['bell']}  Alerts:       `All enabled`\n\n"
        f"Tap a setting below to modify."
    )


def format_standings(standings: list[dict]) -> str:
    """Format MLB standings."""
    if not standings:
        return f"{E['x']} Could not load standings."

    lines = [
        f"{E['trophy']} *MLB Standings*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    ]

    current_div = ""
    for team in standings:
        div = team.get("division", "")
        if div != current_div:
            current_div = div
            short_div = div.replace("American League", "AL").replace("National League", "NL")
            lines.append(f"\n*{short_div}*")
            lines.append(f"```{'Team':<5} {'W':>3}-{'L':>3}  {'PCT':>5}  {'GB':>4}```")

        abbr = TEAM_ABBREV.get(team.get("team", ""), "???")
        lines.append(
            f"```{abbr:<5} {team['wins']:>3}-{team['losses']:>3}  "
            f"{team['pct']:>5}  {str(team['gb']):>4}```"
        )

    return "\n".join(lines)


def format_live_alert(game: dict, alert_type: str,
                      details: str) -> str:
    """Format a live in-game alert."""
    away = game.get("away_abbrev", "???")
    home = game.get("home_abbrev", "???")
    score = f"{game.get('away_score', 0)}-{game.get('home_score', 0)}"

    emoji = {
        "line_move": E["chart"],
        "injury":    E["alert"],
        "edge":      E["diamond"],
        "momentum":  E["bolt"],
    }.get(alert_type, E["bell"])

    return (
        f"{emoji} *Live Alert* {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{E['vs']} `{away}` {score} `{home}`\n\n"
        f"{details}"
    )


def format_line_movement(market: dict, movement: dict) -> str:
    """Format line movement alert."""
    direction = E["up"] if movement["direction"] == "up" else E["down"]
    return (
        f"{E['chart']} *Line Movement*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{E['mag']} {market.get('question', 'Market')}\n\n"
        f"  Open:    `{movement.get('open', 0):.3f}`\n"
        f"  Prev:    `{movement.get('previous', 0):.3f}`\n"
        f"  Current: `{movement.get('current', 0):.3f}` {direction}\n"
        f"  Change:  `{movement.get('change_last', 0):+.1f}%`\n"
        f"  Since open: `{movement.get('change_open', 0):+.1f}%`"
    )


def format_pitcher_report(pitcher: dict, statcast: dict = None) -> str:
    """Format a detailed pitcher report."""
    name = pitcher.get("name", "Unknown")
    stats = pitcher.get("season", pitcher.get("stats", {}).get("pitching", {}))

    lines = [
        f"{E['target']} *Pitcher Report: {name}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    ]

    if stats:
        lines.append(f"```")
        lines.append(f"{'ERA:':<12} {stats.get('era', 'N/A'):>8}")
        lines.append(f"{'WHIP:':<12} {stats.get('whip', 'N/A'):>8}")
        lines.append(f"{'W-L:':<12} {stats.get('wins', 0):>3}-{stats.get('losses', 0)}")
        lines.append(f"{'IP:':<12} {stats.get('inningsPitched', 'N/A'):>8}")
        lines.append(f"{'K/9:':<12} {stats.get('strikeoutsPer9Inn', 'N/A'):>8}")
        lines.append(f"{'BB/9:':<12} {stats.get('walksPer9Inn', 'N/A'):>8}")
        lines.append(f"{'HR/9:':<12} {stats.get('homeRunsPer9', 'N/A'):>8}")
        lines.append(f"{'K:':<12} {stats.get('strikeOuts', 'N/A'):>8}")
        lines.append(f"{'BB:':<12} {stats.get('baseOnBalls', 'N/A'):>8}")
        lines.append(f"```")
    else:
        lines.append(f"{E['yellow']} No season stats available.")

    if statcast and statcast.get("available"):
        lines.append(f"\n{E['chart']} *Statcast Metrics*")
        for key in ["xba", "xslg", "xwoba", "barrel_batted_rate",
                     "hard_hit_percent", "whiff_percent"]:
            val = statcast.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                lines.append(f"  {label}: `{val}`")

    return "\n".join(lines)


def format_h2h(data: dict) -> str:
    """Format head-to-head matchup data."""
    if not data.get("available"):
        return f"{E['x']} No head-to-head data available."

    lines = [
        f"{E['vs']} *Head-to-Head: {data['team1']} vs {data['team2']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Games played: `{data['games']}`\n"
        f"{data['team1']}: `{data['team1_wins']}W`\n"
        f"{data['team2']}: `{data['team2_wins']}W`\n\n"
    ]

    if data.get("matchups"):
        lines.append("*Recent Results*")
        for m in data["matchups"][-5:]:
            lines.append(
                f"  {m['date']}: {m['away_team'][:3]} "
                f"{m['away_score']}-{m['home_score']} "
                f"{m['home_team'][:3]}"
            )

    return "\n".join(lines)


def format_recent_form(data: dict) -> str:
    """Format team recent form."""
    if not data.get("available"):
        return f"{E['x']} Recent form data not available."

    streaks = ""
    for r in data.get("results", [])[-10:]:
        streaks += E["green"] if r["won"] else E["red"]

    return (
        f"{E['fire']} *Recent Form: {data['team']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Last {data['games']} games: "
        f"`{data['wins']}W-{data['losses']}L` "
        f"({data['win_pct']:.0%})\n\n"
        f"Streak: {streaks}\n\n"
        f"Avg Runs For:     `{data['avg_runs_for']:.1f}`\n"
        f"Avg Runs Against: `{data['avg_runs_ag']:.1f}`\n"
        f"Run Diff:         `{data['run_diff']:+.1f}`"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_bar(value: float, max_val: float = 20, width: int = 8) -> str:
    """Make a simple progress bar from emoji."""
    filled = int(min(value / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _confidence_badge(confidence: float) -> str:
    """Return a confidence badge string."""
    if confidence >= 0.7:
        return f"{E['green']} High Confidence ({confidence:.0%})"
    elif confidence >= 0.4:
        return f"{E['yellow']} Medium Confidence ({confidence:.0%})"
    else:
        return f"{E['red']} Low Confidence ({confidence:.0%})"
