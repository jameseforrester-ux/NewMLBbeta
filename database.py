"""
SQLite database layer — tracks bets, P&L, and user settings.
"""

import aiosqlite
import json
from datetime import datetime

DB_PATH = "mlb_edge.db"


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                game_id     TEXT,
                game_desc   TEXT,
                market_type TEXT,
                position    TEXT,
                model_prob  REAL,
                market_prob REAL,
                edge_pct    REAL,
                kelly_size  REAL,
                stake       REAL,
                odds        REAL,
                status      TEXT    DEFAULT 'pending',
                result      TEXT,
                pnl         REAL    DEFAULT 0.0,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_at    TEXT NOT NULL,
                alert_type TEXT,
                game_id    TEXT,
                message    TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_digest (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                date       TEXT NOT NULL,
                picks_json TEXT,
                sent       INTEGER DEFAULT 0
            );
        """)
        await db.commit()


# ── Settings helpers ─────────────────────────────────────────────────────────

async def get_setting(key: str, default: str = "") -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row[0] if row else default


async def set_setting(key: str, value: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        await db.commit()


async def get_bankroll() -> float:
    return float(await get_setting("bankroll", "1000.0"))


async def set_bankroll(amount: float):
    await set_setting("bankroll", str(amount))


async def get_min_edge() -> float:
    return float(await get_setting("min_edge", "6.0"))


async def set_min_edge(pct: float):
    await set_setting("min_edge", str(pct))


async def get_kelly_fraction() -> float:
    return float(await get_setting("kelly_fraction", "0.5"))


async def set_kelly_fraction(frac: float):
    await set_setting("kelly_fraction", str(frac))


# ── Bet tracking ─────────────────────────────────────────────────────────────

async def log_bet(
    game_id: str, game_desc: str, market_type: str, position: str,
    model_prob: float, market_prob: float, edge_pct: float,
    kelly_size: float, stake: float, odds: float, notes: str = ""
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO bets
               (created_at, game_id, game_desc, market_type, position,
                model_prob, market_prob, edge_pct, kelly_size, stake, odds, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(), game_id, game_desc,
                market_type, position, model_prob, market_prob,
                edge_pct, kelly_size, stake, odds, notes,
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def update_bet_result(bet_id: int, result: str, pnl: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE bets SET status = 'settled', result = ?, pnl = ? WHERE id = ?",
            (result, pnl, bet_id),
        )
        await db.commit()


async def get_pending_bets():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM bets WHERE status = 'pending' ORDER BY created_at DESC"
        )
        return await cursor.fetchall()


async def get_bet_history(limit: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM bets ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return await cursor.fetchall()


async def get_pnl_summary():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT
                COUNT(*)                                           AS total_bets,
                SUM(CASE WHEN result = 'win'  THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END)  AS losses,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                COALESCE(SUM(pnl), 0.0)                            AS total_pnl,
                COALESCE(SUM(stake), 0.0)                           AS total_staked,
                COALESCE(AVG(edge_pct), 0.0)                        AS avg_edge
            FROM bets
        """)
        row = await cursor.fetchone()
        keys = ["total_bets", "wins", "losses", "pending",
                "total_pnl", "total_staked", "avg_edge"]
        return dict(zip(keys, row))


# ── Alert log ────────────────────────────────────────────────────────────────

async def log_alert(alert_type: str, game_id: str, message: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO alerts_log (sent_at, alert_type, game_id, message) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), alert_type, game_id, message),
        )
        await db.commit()
