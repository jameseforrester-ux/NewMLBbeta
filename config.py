"""
Configuration & environment loader for MLB Edge Bot.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── External APIs ────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── MLB API (free, no key) ───────────────────────────────────────────────────
MLB_STATS_BASE   = "https://statsapi.mlb.com/api/v1"
MLB_LIVE_BASE    = "https://statsapi.mlb.com/api/v1.1"
SAVANT_BASE      = "https://baseballsavant.mlb.com"
STATCAST_SEARCH  = f"{SAVANT_BASE}/statcast_search/csv"

# ── Polymarket ───────────────────────────────────────────────────────────────
POLY_GAMMA_API   = "https://gamma-api.polymarket.com"
POLY_CLOB_API    = "https://clob.polymarket.com"

# ── The Odds API ─────────────────────────────────────────────────────────────
ODDS_API_BASE    = "https://api.the-odds-api.com/v4"

# ── Betting Defaults ────────────────────────────────────────────────────────
DEFAULT_BANKROLL      = float(os.getenv("BANKROLL", "1000.0"))
DEFAULT_MIN_EDGE      = float(os.getenv("MIN_EDGE_PCT", "6.0"))
DEFAULT_KELLY_FRAC    = float(os.getenv("KELLY_FRACTION", "0.5"))
DIGEST_HOUR           = int(os.getenv("DIGEST_HOUR", "9"))
DIGEST_MINUTE         = int(os.getenv("DIGEST_MINUTE", "0"))
TIMEZONE              = os.getenv("TIMEZONE", "America/Vancouver")

# ── Model Weights (tunable) ─────────────────────────────────────────────────
MODEL_WEIGHTS = {
    "pitcher_era":       0.12,
    "pitcher_fip":       0.14,
    "pitcher_whip":      0.08,
    "pitcher_k9":        0.06,
    "pitcher_bb9":       0.05,
    "team_woba":         0.12,
    "team_ops":          0.08,
    "team_wrc_plus":     0.10,
    "home_away_split":   0.06,
    "recent_form":       0.08,
    "bullpen_era":       0.06,
    "park_factor":       0.05,
}

# ── Park Factors (run environment multiplier — 1.0 = neutral) ───────────────
PARK_FACTORS = {
    "Coors Field":              1.28,
    "Globe Life Field":         1.10,
    "Fenway Park":              1.08,
    "Great American Ball Park": 1.07,
    "Yankee Stadium":           1.05,
    "Citizens Bank Park":       1.04,
    "Wrigley Field":            1.03,
    "Guaranteed Rate Field":    1.02,
    "Angel Stadium":            1.01,
    "Minute Maid Park":         1.01,
    "Rogers Centre":            1.00,
    "Busch Stadium":            0.99,
    "Target Field":             0.99,
    "Dodger Stadium":           0.98,
    "Chase Field":              0.98,
    "Tropicana Field":          0.97,
    "PNC Park":                 0.97,
    "Kauffman Stadium":         0.96,
    "T-Mobile Park":            0.96,
    "Oracle Park":              0.95,
    "Petco Park":               0.94,
    "Citi Field":               0.94,
    "loanDepot park":           0.95,
    "Truist Park":              0.99,
    "Nationals Park":           1.00,
    "American Family Field":    1.02,
    "Comerica Park":            0.96,
    "Progressive Field":        0.98,
    "Oakland Coliseum":         0.95,
    "Camden Yards":             1.01,
}

# ── Team Abbreviation Map ───────────────────────────────────────────────────
TEAM_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

# ── Emoji Constants ──────────────────────────────────────────────────────────
E = {
    "fire":     "🔥", "chart":    "📊", "money":   "💰",
    "alert":    "🚨", "diamond":  "💎", "base":    "⚾",
    "green":    "🟢", "yellow":   "🟡", "red":     "🔴",
    "trophy":   "🏆", "brain":    "🧠", "target":  "🎯",
    "clock":    "⏰", "up":       "📈", "down":    "📉",
    "star":     "⭐", "bolt":     "⚡", "check":   "✅",
    "x":        "❌", "wave":     "👋", "gear":    "⚙️",
    "bell":     "🔔", "bat":      "🏏", "muscle":  "💪",
    "mag":      "🔍", "calendar": "📅", "memo":    "📝",
    "lock":     "🔒", "unlock":   "🔓", "park":    "🏟️",
    "dice":     "🎲", "edge":     "🗡️", "bank":    "🏦",
    "refresh":  "🔄", "home":     "🏠", "away":    "✈️",
    "vs":       "⚔️", "pin":      "📌", "run":     "🏃",
}
