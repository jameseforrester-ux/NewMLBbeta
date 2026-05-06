# ⚾ MLB Edge Bot

> AI-powered MLB betting edge detection for Polymarket.  
> Analyzes Statcast, MLB Stats API, and Polymarket lines to find +EV positions.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## What It Does

MLB Edge Bot is a Telegram bot that:

- **Scans** today's MLB games and pulls pitcher/team stats from MLB Stats API & Baseball Savant
- **Reads** Polymarket MLB markets (moneyline, over/under, run line, NRFI/YRFI)
- **Compares** model-derived probabilities against market-implied odds
- **Finds edges** where the market is mispricing outcomes (≥6% by default)
- **Sizes positions** using Kelly Criterion with configurable bankroll
- **Sends alerts** — daily digest, live in-game edges, line movements, injury news
- **Tracks** all bets, P&L, and win rate in a local SQLite database

### Model Components

| Factor | Weight | Source |
|--------|--------|--------|
| Starting pitcher ERA/FIP | 26% | MLB Stats API |
| Starting pitcher WHIP/K9/BB9 | 19% | MLB Stats API + Statcast |
| Team batting (OPS/wOBA/wRC+) | 30% | MLB Stats API |
| Bullpen ERA | 6% | MLB Stats API |
| Home/away splits | 6% | Historical + recent |
| Recent form (L14) | 8% | MLB Stats API |
| Park factor | 5% | Built-in lookup |

Win probability uses a **Log5 + composite weighted blend**. Expected runs use pitcher/batter matchup quality scaled by park factor. First-inning scoring uses pitcher quality adjustments against the ~50% MLB base rate.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+ on your VPS
- A Telegram bot (create one via [@BotFather](https://t.me/BotFather))
- (Optional) Free API key from [The Odds API](https://the-odds-api.com)

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mlb-edge-bot.git
cd mlb-edge-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
nano .env
```

Fill in your values:

```env
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_CHAT_ID=your_chat_id
BANKROLL=1000.00
MIN_EDGE_PCT=6.0
KELLY_FRACTION=0.5
TIMEZONE=America/Vancouver
```

**How to find your Chat ID:** Message [@userinfobot](https://t.me/userinfobot) on Telegram.

### 4. Run

```bash
python bot.py
```

For persistent running on your VPS:

```bash
# Using screen
screen -S mlb-bot
python bot.py
# Ctrl+A, D to detach

# Or using systemd (recommended)
sudo nano /etc/systemd/system/mlb-edge-bot.service
```

Systemd service file:

```ini
[Unit]
Description=MLB Edge Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/mlb-edge-bot
ExecStart=/path/to/mlb-edge-bot/venv/bin/python bot.py
Restart=always
RestartSec=10
EnvironmentFile=/path/to/mlb-edge-bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mlb-edge-bot
sudo systemctl start mlb-edge-bot
sudo journalctl -u mlb-edge-bot -f  # view logs
```

---

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Main menu with all navigation |
| `/games` | Today's full game slate |
| `/scan` | Scan all games for edges |
| `/pnl` | Performance dashboard |
| `/standings` | Current MLB standings |
| `/settings` | Configure bankroll, edge, Kelly |
| `/help` | Command reference |

### Menu Features

- **Today's Games** — Full slate with pitchers, venues, live scores
- **Find Edges** — Scan all games against Polymarket lines
- **Markets** — Browse active Polymarket MLB markets
- **Research Center** — Pitcher lookup, team stats, H2H, recent form
- **P&L Dashboard** — Track record, ROI, total staked
- **Bet History** — Log of all bets placed
- **Settings** — Bankroll, minimum edge threshold, Kelly fraction
- **Alert Controls** — Toggle digest, live, line movement, injury alerts

### Inline Navigation

Every screen has inline keyboard buttons for seamless navigation. Tap any game to get:
- Full statistical analysis
- Edge detection
- Pitcher reports with Statcast metrics
- Head-to-head history
- Recent form (last 14 games)
- Matching Polymarket markets

---

## Alert System

| Alert | Frequency | Trigger |
|-------|-----------|---------|
| Daily Digest | Once daily (configurable) | Scheduled |
| Live Edge | Every 2 minutes | In-game mispricing |
| Line Movement | Every 2 minutes | ≥3% price swing |
| Injury/Lineup | Every 15 minutes | TBD pitcher, roster changes |

---

## Project Structure

```
mlb-edge-bot/
├── bot.py           # Main bot — commands, menus, callbacks
├── config.py        # Environment, constants, park factors
├── database.py      # SQLite — bets, P&L, settings
├── mlb_data.py      # MLB Stats API + Baseball Savant data
├── polymarket.py    # Polymarket Gamma + CLOB API
├── model.py         # Edge detection, Kelly, probability model
├── formatters.py    # Rich Telegram message formatting
├── alerts.py        # Scheduled digests, live monitor, injury check
├── requirements.txt
├── .env.example
└── README.md
```

---

## How Edge Detection Works

```
1. Fetch game data (pitchers, team stats, form, H2H, park)
        ↓
2. Score each component 0-1 using weighted model
        ↓
3. Calculate win probability (Log5 + composite blend)
        ↓
4. Calculate expected total runs & first-inning probability
        ↓
5. Fetch Polymarket implied odds from market prices
        ↓
6. Edge = Model Probability - Market Implied Probability
        ↓
7. If Edge ≥ threshold → Kelly Criterion → Position size → Alert
```

---

## Important Notes

⚠️ **This bot is a tool, not a guarantee.** No model beats the market consistently at 80%. What it does is identify +EV spots where the data suggests mispricing. Over a large enough sample, +EV betting is profitable — but individual bets can and will lose.

- **Start with small stakes** and track your results
- **Half-Kelly is the default** because full Kelly is extremely volatile
- The **6% edge minimum** filters for higher-quality plays
- **Park factors and pitcher stats** are the strongest predictors in the model
- Markets become more efficient closer to game time — early edges may close

---

## License

MIT — use it, modify it, profit from it.
