"""
MLB Edge Bot — Main Telegram bot with full menu system.

Run: python bot.py
"""

import asyncio
import logging
from datetime import time as dtime
from zoneinfo import ZoneInfo

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    CallbackQuery,
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters,
)
from telegram.constants import ParseMode

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    DIGEST_HOUR, DIGEST_MINUTE, TIMEZONE,
    DEFAULT_BANKROLL, DEFAULT_MIN_EDGE, DEFAULT_KELLY_FRAC, E,
)
from database import (
    init_db, get_bankroll, set_bankroll,
    get_min_edge, set_min_edge,
    get_kelly_fraction, set_kelly_fraction,
    get_pnl_summary, log_bet, get_bet_history,
    get_pending_bets, get_setting, set_setting,
)
from mlb_data import (
    get_todays_schedule, get_live_game,
    get_pitcher_detailed, get_team_recent_form,
    get_head_to_head, get_standings, search_team_id,
    get_team_stats, get_player_stats, get_pitcher_statcast,
)
from polymarket import (
    get_todays_mlb_markets, get_market_implied_odds,
    search_mlb_markets,
)
from model import (
    analyze_game, find_edges, kelly_criterion,
    calculate_stake, generate_reasoning,
)
from formatters import (
    format_main_menu, format_games_list, format_game_card,
    format_analysis, format_edge_alert, format_daily_digest,
    format_pnl, format_settings, format_standings,
    format_pitcher_report, format_h2h, format_recent_form,
    format_live_alert, format_line_movement,
)
from alerts import (
    run_daily_digest, run_live_monitor, run_injury_check,
    _analyze_single_game,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")


# ═══════════════════════════════════════════════════════════════════════════
#  KEYBOARD LAYOUTS
# ═══════════════════════════════════════════════════════════════════════════

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{E['calendar']} Today's Games", callback_data="games"),
            InlineKeyboardButton(f"{E['fire']} Find Edges", callback_data="scan"),
        ],
        [
            InlineKeyboardButton(f"{E['chart']} Markets", callback_data="markets"),
            InlineKeyboardButton(f"{E['trophy']} Standings", callback_data="standings"),
        ],
        [
            InlineKeyboardButton(f"{E['mag']} Research", callback_data="research"),
            InlineKeyboardButton(f"{E['money']} P&L", callback_data="pnl"),
        ],
        [
            InlineKeyboardButton(f"{E['bell']} Alerts", callback_data="alerts_menu"),
            InlineKeyboardButton(f"{E['gear']} Settings", callback_data="settings"),
        ],
        [
            InlineKeyboardButton(f"{E['memo']} Bet History", callback_data="history"),
            InlineKeyboardButton(f"{E['refresh']} Refresh", callback_data="refresh"),
        ],
    ])


def games_keyboard(games: list[dict]):
    buttons = []
    for g in games:
        away = g.get("away_abbrev", "?")
        home = g.get("home_abbrev", "?")
        status = ""
        if "In Progress" in g.get("status", ""):
            status = f" {E['green']}"
        elif "Final" in g.get("status", ""):
            status = f" {E['check']}"

        buttons.append([InlineKeyboardButton(
            f"{away} @ {home}{status}",
            callback_data=f"game_{g.get('game_id')}"
        )])

    buttons.append([InlineKeyboardButton(
        f"{E['fire']} Scan All for Edges",
        callback_data="scan"
    )])
    buttons.append([InlineKeyboardButton(
        f"« Back to Menu",
        callback_data="menu"
    )])
    return InlineKeyboardMarkup(buttons)


def game_detail_keyboard(game_id):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{E['brain']} Full Analysis", callback_data=f"analyze_{game_id}"),
            InlineKeyboardButton(f"{E['edge']} Find Edge", callback_data=f"edge_{game_id}"),
        ],
        [
            InlineKeyboardButton(f"{E['target']} Pitchers", callback_data=f"pitchers_{game_id}"),
            InlineKeyboardButton(f"{E['vs']} H2H", callback_data=f"h2h_{game_id}"),
        ],
        [
            InlineKeyboardButton(f"{E['fire']} Form", callback_data=f"form_{game_id}"),
            InlineKeyboardButton(f"{E['chart']} Markets", callback_data=f"gmarkets_{game_id}"),
        ],
        [InlineKeyboardButton("« Back to Games", callback_data="games")],
    ])


def research_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{E['target']} Pitcher Lookup", callback_data="lookup_pitcher"),
            InlineKeyboardButton(f"{E['bat']} Team Stats", callback_data="lookup_team"),
        ],
        [
            InlineKeyboardButton(f"{E['vs']} H2H Matchup", callback_data="lookup_h2h"),
            InlineKeyboardButton(f"{E['fire']} Team Form", callback_data="lookup_form"),
        ],
        [
            InlineKeyboardButton(f"{E['trophy']} Standings", callback_data="standings"),
        ],
        [InlineKeyboardButton("« Back to Menu", callback_data="menu")],
    ])


def settings_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{E['bank']} Set Bankroll", callback_data="set_bankroll")],
        [InlineKeyboardButton(f"{E['edge']} Set Min Edge", callback_data="set_edge")],
        [InlineKeyboardButton(f"{E['dice']} Set Kelly Fraction", callback_data="set_kelly")],
        [InlineKeyboardButton(f"{E['bell']} Alert Settings", callback_data="alerts_menu")],
        [InlineKeyboardButton("« Back to Menu", callback_data="menu")],
    ])


def alerts_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{E['calendar']} Send Digest Now", callback_data="digest_now")],
        [InlineKeyboardButton(f"{E['bell']} Toggle Live Alerts", callback_data="toggle_live")],
        [InlineKeyboardButton(f"{E['chart']} Toggle Line Alerts", callback_data="toggle_lines")],
        [InlineKeyboardButton(f"{E['alert']} Toggle Injury News", callback_data="toggle_injury")],
        [InlineKeyboardButton("« Back to Menu", callback_data="menu")],
    ])


def back_keyboard(target: str = "menu"):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("« Back", callback_data=target)],
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await init_db()

    # Initialize defaults if not set
    if not await get_setting("bankroll"):
        await set_bankroll(DEFAULT_BANKROLL)
    if not await get_setting("min_edge"):
        await set_min_edge(DEFAULT_MIN_EDGE)
    if not await get_setting("kelly_fraction"):
        await set_kelly_fraction(DEFAULT_KELLY_FRAC)

    text = format_main_menu()
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def cmd_games(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /games command."""
    await _show_games(update.message, context)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scan command — full edge scan."""
    await _run_edge_scan(update.message, context)


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pnl command."""
    await _show_pnl(update.message, context)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /settings command."""
    await _show_settings(update.message, context)


async def cmd_standings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /standings command."""
    await _show_standings(update.message, context)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    text = (
        f"{E['base']} *MLB Edge Bot — Commands*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"/start — Main menu\n"
        f"/games — Today's games\n"
        f"/scan — Scan all games for edges\n"
        f"/pnl — Performance dashboard\n"
        f"/standings — MLB standings\n"
        f"/settings — Bot settings\n"
        f"/help — This message\n\n"
        f"{E['mag']} *Quick Research*\n"
        f"Send a team name to get a quick report.\n"
        f"Example: `Yankees` or `LAD`\n\n"
        f"{E['bell']} *Auto Alerts*\n"
        f"Daily digest sent at {DIGEST_HOUR}:{DIGEST_MINUTE:02d}\n"
        f"Live game alerts every 2 min\n"
        f"Line movement alerts on 3%+ swings"
    )
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACK QUERY HANDLER (menu navigation)
# ═══════════════════════════════════════════════════════════════════════════

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data

    try:
        if data == "menu":
            await _edit_or_send(query, format_main_menu(), main_menu_keyboard())

        elif data == "games":
            await _show_games(query, context)

        elif data.startswith("game_"):
            game_id = int(data.split("_", 1)[1])
            await _show_game_detail(query, context, game_id)

        elif data.startswith("analyze_"):
            game_id = int(data.split("_", 1)[1])
            await _show_analysis(query, context, game_id)

        elif data.startswith("edge_"):
            game_id = int(data.split("_", 1)[1])
            await _show_game_edge(query, context, game_id)

        elif data.startswith("pitchers_"):
            game_id = int(data.split("_", 1)[1])
            await _show_pitchers(query, context, game_id)

        elif data.startswith("h2h_"):
            game_id = int(data.split("_", 1)[1])
            await _show_h2h(query, context, game_id)

        elif data.startswith("form_"):
            game_id = int(data.split("_", 1)[1])
            await _show_form(query, context, game_id)

        elif data.startswith("gmarkets_"):
            game_id = int(data.split("_", 1)[1])
            await _show_game_markets(query, context, game_id)

        elif data == "scan":
            await _run_edge_scan(query, context)

        elif data == "markets":
            await _show_markets(query, context)

        elif data == "standings":
            await _show_standings(query, context)

        elif data == "research":
            text = (
                f"{E['mag']} *Research Center*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Deep-dive into matchup data.\n"
                f"Select a research tool below."
            )
            await _edit_or_send(query, text, research_keyboard())

        elif data == "pnl":
            await _show_pnl(query, context)

        elif data == "history":
            await _show_history(query, context)

        elif data == "settings":
            await _show_settings(query, context)

        elif data == "set_bankroll":
            context.user_data["awaiting"] = "bankroll"
            await _edit_or_send(
                query,
                f"{E['bank']} *Set Bankroll*\n\n"
                f"Send your total bankroll amount.\n"
                f"Example: `1000` or `500.50`",
                back_keyboard("settings"),
            )

        elif data == "set_edge":
            context.user_data["awaiting"] = "min_edge"
            await _edit_or_send(
                query,
                f"{E['edge']} *Set Minimum Edge*\n\n"
                f"Send minimum edge % to flag.\n"
                f"Example: `6` for 6%",
                back_keyboard("settings"),
            )

        elif data == "set_kelly":
            context.user_data["awaiting"] = "kelly"
            await _edit_or_send(
                query,
                f"{E['dice']} *Set Kelly Fraction*\n\n"
                f"Send fraction of full Kelly (0.1 - 1.0).\n"
                f"`0.5` = Half Kelly (recommended)\n"
                f"`0.25` = Quarter Kelly (conservative)\n"
                f"`1.0` = Full Kelly (aggressive)",
                back_keyboard("settings"),
            )

        elif data == "alerts_menu":
            live_on = await get_setting("alerts_live", "on")
            lines_on = await get_setting("alerts_lines", "on")
            injury_on = await get_setting("alerts_injury", "on")

            text = (
                f"{E['bell']} *Alert Settings*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{E['calendar']} Daily Digest: `Always on`\n"
                f"  Sent at {DIGEST_HOUR}:{DIGEST_MINUTE:02d} {TIMEZONE}\n\n"
                f"{E['green'] if live_on == 'on' else E['red']} Live Game Alerts: `{live_on}`\n"
                f"{E['green'] if lines_on == 'on' else E['red']} Line Movement: `{lines_on}`\n"
                f"{E['green'] if injury_on == 'on' else E['red']} Injury/Lineup: `{injury_on}`"
            )
            await _edit_or_send(query, text, alerts_keyboard())

        elif data == "digest_now":
            # Manually trigger digest
            context.job_queue.run_once(
                run_daily_digest, 0,
                data={"chat_id": query.message.chat_id},
            )
            await query.message.reply_text(
                f"{E['refresh']} Running digest scan now..."
            )

        elif data.startswith("toggle_"):
            alert_type = data.replace("toggle_", "")
            key = f"alerts_{alert_type}"
            current = await get_setting(key, "on")
            new_val = "off" if current == "on" else "on"
            await set_setting(key, new_val)
            status = E['green'] if new_val == 'on' else E['red']
            await query.message.reply_text(
                f"{status} {alert_type.title()} alerts: `{new_val}`",
                parse_mode=ParseMode.MARKDOWN,
            )

        elif data == "refresh":
            await _edit_or_send(query, format_main_menu(), main_menu_keyboard())

        elif data.startswith("lookup_"):
            lookup_type = data.replace("lookup_", "")
            context.user_data["awaiting"] = f"lookup_{lookup_type}"
            prompts = {
                "pitcher": "Send pitcher name (e.g. `Gerrit Cole`)",
                "team":    "Send team name (e.g. `Yankees` or `NYY`)",
                "h2h":     "Send two teams separated by 'vs'\n(e.g. `Yankees vs Red Sox`)",
                "form":    "Send team name (e.g. `Dodgers`)",
            }
            await _edit_or_send(
                query,
                f"{E['mag']} *{lookup_type.title()} Lookup*\n\n{prompts.get(lookup_type, 'Send query')}",
                back_keyboard("research"),
            )

    except Exception as e:
        log.error(f"Callback error [{data}]: {e}")
        await query.message.reply_text(
            f"{E['x']} Error: {str(e)[:200]}",
            reply_markup=back_keyboard("menu"),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT MESSAGE HANDLER (for settings input & lookups)
# ═══════════════════════════════════════════════════════════════════════════

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle free-text input (settings values, lookups)."""
    text = update.message.text.strip()
    awaiting = context.user_data.get("awaiting")

    if awaiting == "bankroll":
        try:
            amount = float(text.replace("$", "").replace(",", ""))
            await set_bankroll(amount)
            context.user_data.pop("awaiting", None)
            await update.message.reply_text(
                f"{E['check']} Bankroll set to `${amount:.2f}`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard("settings"),
            )
        except ValueError:
            await update.message.reply_text(
                f"{E['x']} Invalid amount. Send a number like `1000`",
                parse_mode=ParseMode.MARKDOWN,
            )

    elif awaiting == "min_edge":
        try:
            edge = float(text.replace("%", ""))
            await set_min_edge(edge)
            context.user_data.pop("awaiting", None)
            await update.message.reply_text(
                f"{E['check']} Minimum edge set to `{edge:.1f}%`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard("settings"),
            )
        except ValueError:
            await update.message.reply_text(
                f"{E['x']} Invalid value. Send a number like `6`",
                parse_mode=ParseMode.MARKDOWN,
            )

    elif awaiting == "kelly":
        try:
            frac = float(text)
            if not 0 < frac <= 1:
                raise ValueError
            await set_kelly_fraction(frac)
            context.user_data.pop("awaiting", None)
            await update.message.reply_text(
                f"{E['check']} Kelly fraction set to `{frac:.0%}`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard("settings"),
            )
        except ValueError:
            await update.message.reply_text(
                f"{E['x']} Send a value between 0.1 and 1.0",
                parse_mode=ParseMode.MARKDOWN,
            )

    elif awaiting == "lookup_form":
        context.user_data.pop("awaiting", None)
        await update.message.reply_text(f"{E['refresh']} Loading form data...")
        form = await get_team_recent_form(text)
        msg = format_recent_form(form)
        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard("research"),
        )

    elif awaiting == "lookup_h2h":
        context.user_data.pop("awaiting", None)
        parts = text.lower().replace(" vs ", "|").replace(" v ", "|").split("|")
        if len(parts) != 2:
            await update.message.reply_text(
                f"{E['x']} Format: `Team1 vs Team2`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        await update.message.reply_text(f"{E['refresh']} Loading H2H data...")
        h2h = await get_head_to_head(parts[0].strip(), parts[1].strip())
        msg = format_h2h(h2h)
        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard("research"),
        )

    elif awaiting == "lookup_pitcher":
        context.user_data.pop("awaiting", None)
        await update.message.reply_text(f"{E['refresh']} Looking up pitcher...")
        # Search by name in today's games first
        games = await get_todays_schedule()
        pitcher_id = None
        for g in games:
            for side in ["home", "away"]:
                p = g.get(f"{side}_pitcher", {})
                if text.lower() in p.get("name", "").lower():
                    pitcher_id = p.get("id")
                    break
            if pitcher_id:
                break

        if pitcher_id:
            stats = await get_pitcher_detailed(pitcher_id)
            statcast = await get_pitcher_statcast(pitcher_id)
            msg = format_pitcher_report(stats, statcast)
        else:
            msg = f"{E['x']} Pitcher not found in today's scheduled starters.\nTry a name like `Gerrit Cole`."

        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard("research"),
        )

    elif awaiting == "lookup_team":
        context.user_data.pop("awaiting", None)
        await update.message.reply_text(f"{E['refresh']} Loading team stats...")
        team_id = await search_team_id(text)
        if team_id:
            stats = await get_team_stats(team_id)
            hitting = stats.get("hitting", {})
            pitching = stats.get("pitching", {})
            msg = (
                f"{E['chart']} *Team Stats: {text}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"*Hitting*\n"
                f"```\n"
                f"AVG:  {hitting.get('avg', 'N/A'):>8}\n"
                f"OBP:  {hitting.get('obp', 'N/A'):>8}\n"
                f"SLG:  {hitting.get('slg', 'N/A'):>8}\n"
                f"OPS:  {hitting.get('ops', 'N/A'):>8}\n"
                f"HR:   {hitting.get('homeRuns', 'N/A'):>8}\n"
                f"R:    {hitting.get('runs', 'N/A'):>8}\n"
                f"RBI:  {hitting.get('rbi', 'N/A'):>8}\n"
                f"SB:   {hitting.get('stolenBases', 'N/A'):>8}\n"
                f"```\n\n"
                f"*Pitching*\n"
                f"```\n"
                f"ERA:  {pitching.get('era', 'N/A'):>8}\n"
                f"WHIP: {pitching.get('whip', 'N/A'):>8}\n"
                f"K:    {pitching.get('strikeOuts', 'N/A'):>8}\n"
                f"BB:   {pitching.get('baseOnBalls', 'N/A'):>8}\n"
                f"SV:   {pitching.get('saves', 'N/A'):>8}\n"
                f"```"
            )
        else:
            msg = f"{E['x']} Team not found. Try full name or abbreviation."

        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard("research"),
        )

    else:
        # Default: treat as a quick team lookup
        await update.message.reply_text(
            f"{E['refresh']} Looking up `{text}`...",
            parse_mode=ParseMode.MARKDOWN,
        )
        form = await get_team_recent_form(text)
        if form.get("available"):
            msg = format_recent_form(form)
            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_menu_keyboard(),
            )
        else:
            await update.message.reply_text(
                f"No data found for `{text}`.\n"
                f"Try a team name like `Dodgers` or use /start for the menu.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_menu_keyboard(),
            )


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL ACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def _show_games(target, context):
    """Display today's games list."""
    games = await get_todays_schedule()
    text = format_games_list(games)
    kb = games_keyboard(games) if games else back_keyboard("menu")
    await _edit_or_send(target, text, kb)


async def _show_game_detail(query: CallbackQuery, context, game_id: int):
    """Show detail view for a specific game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)

    if not game:
        await _edit_or_send(query, f"{E['x']} Game not found.", back_keyboard("games"))
        return

    text = format_game_card(game)
    await _edit_or_send(query, text, game_detail_keyboard(game_id))


async def _show_analysis(query: CallbackQuery, context, game_id: int):
    """Run and show full analysis for a game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        await _edit_or_send(query, f"{E['x']} Game not found.", back_keyboard("games"))
        return

    await query.message.reply_text(f"{E['brain']} Running full analysis...")

    markets = await get_todays_mlb_markets()
    min_edge = await get_min_edge()

    try:
        analysis, edges = await _analyze_single_game(game, markets, min_edge)
        text = format_analysis(game, analysis)

        if edges:
            text += f"\n\n{E['diamond']} *{len(edges)} edge(s) found!*\n"
            for edge in edges[:3]:
                text += f"  {E['target']} {edge['bet_desc']}: +{edge['edge_pct']:.1f}%\n"

        await query.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN,
            reply_markup=game_detail_keyboard(game_id),
        )
    except Exception as e:
        await query.message.reply_text(
            f"{E['x']} Analysis error: {str(e)[:200]}",
            reply_markup=game_detail_keyboard(game_id),
        )


async def _show_game_edge(query: CallbackQuery, context, game_id: int):
    """Find and show edges for a specific game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        await _edit_or_send(query, f"{E['x']} Game not found.", back_keyboard("games"))
        return

    await query.message.reply_text(f"{E['edge']} Scanning for edges...")

    markets = await get_todays_mlb_markets()
    min_edge = await get_min_edge()
    bankroll = await get_bankroll()
    kelly_frac = await get_kelly_fraction()

    try:
        analysis, edges = await _analyze_single_game(game, markets, min_edge)

        if not edges:
            await query.message.reply_text(
                f"{E['mag']} No edges found meeting {min_edge}% threshold.\n"
                f"Market may be efficiently priced for this game.",
                reply_markup=game_detail_keyboard(game_id),
            )
            return

        for edge in edges[:3]:
            kelly = kelly_criterion(edge["model_prob"], edge["price"], kelly_frac)
            stake = calculate_stake(bankroll, kelly["kelly_pct"])
            reasoning = generate_reasoning(analysis, edge, game)
            alert_msg = format_edge_alert(game, edge, analysis, kelly, stake)
            full_msg = f"{alert_msg}\n\n{reasoning}"

            await query.message.reply_text(
                full_msg, parse_mode=ParseMode.MARKDOWN,
                reply_markup=game_detail_keyboard(game_id),
            )
    except Exception as e:
        await query.message.reply_text(
            f"{E['x']} Edge scan error: {str(e)[:200]}",
            reply_markup=game_detail_keyboard(game_id),
        )


async def _show_pitchers(query: CallbackQuery, context, game_id: int):
    """Show pitcher reports for a game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        return

    for side in ["away", "home"]:
        pitcher = game.get(f"{side}_pitcher", {})
        pid = pitcher.get("id")
        if pid:
            stats = await get_pitcher_detailed(pid)
            statcast = await get_pitcher_statcast(pid)
            msg = format_pitcher_report(stats, statcast)
            label = game.get(f"{side}_abbrev", "?")
            await query.message.reply_text(
                f"{E['away'] if side == 'away' else E['home']} *{label} Starter*\n{msg}",
                parse_mode=ParseMode.MARKDOWN,
            )


async def _show_h2h(query: CallbackQuery, context, game_id: int):
    """Show head-to-head for a game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        return

    h2h = await get_head_to_head(
        game.get("home_team", ""),
        game.get("away_team", ""),
    )
    msg = format_h2h(h2h)
    await _edit_or_send(query, msg, game_detail_keyboard(game_id))


async def _show_form(query: CallbackQuery, context, game_id: int):
    """Show recent form for both teams in a game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        return

    for team in [game.get("away_team"), game.get("home_team")]:
        form = await get_team_recent_form(team)
        msg = format_recent_form(form)
        await query.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def _show_game_markets(query: CallbackQuery, context, game_id: int):
    """Show Polymarket markets for a specific game."""
    games = await get_todays_schedule()
    game = next((g for g in games if g.get("game_id") == game_id), None)
    if not game:
        return

    markets = await get_todays_mlb_markets()
    away_kw = game.get("away_team", "").lower().split()[-1]
    home_kw = game.get("home_team", "").lower().split()[-1]

    matching = [
        m for m in markets
        if any(kw in m.get("question", "").lower() for kw in [away_kw, home_kw])
    ]

    if not matching:
        await query.message.reply_text(
            f"{E['x']} No Polymarket markets found for this game.",
            reply_markup=game_detail_keyboard(game_id),
        )
        return

    lines = [
        f"{E['chart']} *Markets: {game.get('away_abbrev')} @ {game.get('home_abbrev')}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    ]

    for market in matching[:6]:
        lines.append(f"\n{E['diamond']} *{market['question'][:60]}*")
        lines.append(f"  Type: `{market['market_type'].upper()}`")
        lines.append(f"  Vol: `${market.get('volume', 0):,.0f}`")
        for outcome in market.get("outcomes", []):
            price = outcome.get("price")
            if price:
                lines.append(f"  {outcome['name']}: `{price:.1%}`")
        lines.append(f"  [View]({market.get('url', '')})")

    await query.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
        reply_markup=game_detail_keyboard(game_id),
    )


async def _run_edge_scan(target, context):
    """Full edge scan across all today's games."""
    msg = await _send_msg(target, f"{E['refresh']} Scanning entire slate for edges...")

    games = await get_todays_schedule()
    markets = await get_todays_mlb_markets()
    bankroll = await get_bankroll()
    min_edge = await get_min_edge()
    kelly_frac = await get_kelly_fraction()

    all_picks = []
    for game in games:
        if game.get("status") not in ("Scheduled", "Pre-Game", "Warmup", "In Progress"):
            continue
        try:
            analysis, edges = await _analyze_single_game(game, markets, min_edge)
            for edge in edges:
                kelly = kelly_criterion(edge["model_prob"], edge["price"], kelly_frac)
                if kelly["should_bet"]:
                    stake = calculate_stake(bankroll, kelly["kelly_pct"])
                    all_picks.append({
                        "game": game, "analysis": analysis,
                        "edge": edge, "kelly": kelly, "stake": stake,
                    })
        except Exception as e:
            log.warning(f"Scan error for {game.get('game_id')}: {e}")

    all_picks.sort(key=lambda p: p["edge"]["edge_pct"], reverse=True)

    digest = format_daily_digest(all_picks)
    await _send_msg(target, digest, ParseMode.MARKDOWN, main_menu_keyboard())

    # Send top 3 detailed
    for pick in all_picks[:3]:
        reasoning = generate_reasoning(pick["analysis"], pick["edge"], pick["game"])
        alert_msg = format_edge_alert(
            pick["game"], pick["edge"],
            pick["analysis"], pick["kelly"], pick["stake"],
        )
        await _send_msg(target, f"{alert_msg}\n\n{reasoning}", ParseMode.MARKDOWN)
        await asyncio.sleep(0.5)


async def _show_markets(target, context):
    """Show all MLB Polymarket markets."""
    markets = await get_todays_mlb_markets()

    if not markets:
        await _send_msg(
            target,
            f"{E['x']} No MLB markets found on Polymarket.",
            reply_markup=back_keyboard("menu"),
        )
        return

    lines = [
        f"{E['chart']} *Polymarket MLB Markets*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{len(markets)} active market(s)\n"
    ]

    for m in markets[:15]:
        type_emoji = {
            "moneyline": E["vs"],
            "over_under": E["chart"],
            "run_line": E["target"],
            "first_inning": E["dice"],
            "prop": E["star"],
        }.get(m["market_type"], E["diamond"])

        lines.append(f"\n{type_emoji} *{m['question'][:55]}*")
        for o in m.get("outcomes", []):
            p = o.get("price")
            if p:
                lines.append(f"  {o['name']}: `{p:.1%}`")

    await _send_msg(
        target,
        "\n".join(lines),
        ParseMode.MARKDOWN,
        back_keyboard("menu"),
    )


async def _show_standings(target, context):
    """Show MLB standings."""
    standings = await get_standings()
    text = format_standings(standings)
    await _send_msg(target, text, ParseMode.MARKDOWN, back_keyboard("menu"))


async def _show_pnl(target, context):
    """Show P&L dashboard."""
    summary = await get_pnl_summary()
    bankroll = await get_bankroll()
    text = format_pnl(summary, bankroll)
    await _send_msg(target, text, ParseMode.MARKDOWN, back_keyboard("menu"))


async def _show_settings(target, context):
    """Show bot settings."""
    bankroll = await get_bankroll()
    min_edge = await get_min_edge()
    kelly_frac = await get_kelly_fraction()
    text = format_settings(bankroll, min_edge, kelly_frac)
    await _edit_or_send(target, text, settings_keyboard())


async def _show_history(target, context):
    """Show bet history."""
    bets = await get_bet_history(15)

    if not bets:
        text = f"{E['memo']} *Bet History*\n━━━━━━━━━━━━━━━━━━━━━━━\n\nNo bets recorded yet."
        await _send_msg(target, text, ParseMode.MARKDOWN, back_keyboard("menu"))
        return

    lines = [f"{E['memo']} *Bet History*\n━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for bet in bets:
        status_emoji = {
            "pending": E["yellow"],
            "win": E["green"],
            "loss": E["red"],
        }.get(bet["result"] or bet["status"], E["yellow"])

        lines.append(
            f"{status_emoji} {bet['game_desc'] or 'N/A'}\n"
            f"  {bet['market_type']}: {bet['position']}\n"
            f"  Edge: `{bet['edge_pct']:.1f}%` | "
            f"Stake: `${bet['stake']:.2f}` | "
            f"P&L: `${bet['pnl']:+.2f}`\n"
        )

    await _send_msg(target, "\n".join(lines), ParseMode.MARKDOWN, back_keyboard("menu"))


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

async def _edit_or_send(target, text: str, keyboard=None):
    """Edit an existing message or send a new one depending on target type."""
    kwargs = {"parse_mode": ParseMode.MARKDOWN}
    if keyboard:
        kwargs["reply_markup"] = keyboard

    if isinstance(target, CallbackQuery):
        try:
            await target.message.edit_text(text, **kwargs)
        except Exception:
            await target.message.reply_text(text, **kwargs)
    else:
        await target.reply_text(text, **kwargs)


async def _send_msg(target, text: str, parse_mode=None, reply_markup=None):
    """Send a message from any target type."""
    kwargs = {}
    if parse_mode:
        kwargs["parse_mode"] = parse_mode
    if reply_markup:
        kwargs["reply_markup"] = reply_markup

    if isinstance(target, CallbackQuery):
        return await target.message.reply_text(text, **kwargs)
    else:
        return await target.reply_text(text, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  APPLICATION SETUP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN in .env file!")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # ── Commands ──────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("games", cmd_games))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("standings", cmd_standings))
    app.add_handler(CommandHandler("help", cmd_help))

    # ── Callbacks (inline keyboard) ──────────────────────────────────
    app.add_handler(CallbackQueryHandler(handle_callback))

    # ── Text messages ────────────────────────────────────────────────
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_text
    ))

    # ── Scheduled jobs ───────────────────────────────────────────────
    tz = ZoneInfo(TIMEZONE)
    chat_id = TELEGRAM_CHAT_ID

    if chat_id:
        jq = app.job_queue

        # Daily digest
        jq.run_daily(
            run_daily_digest,
            time=dtime(hour=DIGEST_HOUR, minute=DIGEST_MINUTE, tzinfo=tz),
            data={"chat_id": int(chat_id)},
            name="daily_digest",
        )

        # Live game monitor (every 2 minutes)
        jq.run_repeating(
            run_live_monitor,
            interval=120,
            first=30,
            data={"chat_id": int(chat_id)},
            name="live_monitor",
        )

        # Injury/lineup check (every 15 minutes)
        jq.run_repeating(
            run_injury_check,
            interval=900,
            first=60,
            data={"chat_id": int(chat_id)},
            name="injury_check",
        )

        log.info(f"Scheduled jobs configured for chat_id={chat_id}")
    else:
        log.warning("No TELEGRAM_CHAT_ID set — scheduled alerts disabled. "
                     "Use /start to interact manually.")

    log.info("MLB Edge Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
