"""
MLB data layer — fetches from MLB Stats API and Baseball Savant / Statcast.
All endpoints are free and require no API key.
"""

import aiohttp
import asyncio
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from typing import Optional
import logging
import json

from config import MLB_STATS_BASE, SAVANT_BASE, TEAM_ABBREV

log = logging.getLogger("mlb_data")

# ── Cache layer (in-memory, TTL-based) ───────────────────────────────────────

_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str):
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.utcnow() - ts).total_seconds() < CACHE_TTL:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val):
    _cache[key] = (val, datetime.utcnow())


# ── HTTP helper ──────────────────────────────────────────────────────────────

async def _fetch_json(url: str, params: dict = None) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
                log.warning(f"HTTP {resp.status} from {url}")
                return {}
    except Exception as e:
        log.error(f"Fetch error {url}: {e}")
        return {}


async def _fetch_text(url: str, params: dict = None) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    return await resp.text()
                return ""
    except Exception as e:
        log.error(f"Fetch error {url}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
#  MLB STATS API
# ═══════════════════════════════════════════════════════════════════════════

async def get_todays_schedule() -> list[dict]:
    """Get today's MLB games with probable pitchers."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"schedule_{today}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        {"sportId": 1, "date": today, "hydrate": "probablePitcher,team,venue,linescore"}
    )

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            game = {
                "game_id":      g.get("gamePk"),
                "game_date":    g.get("gameDate"),
                "status":       g.get("status", {}).get("detailedState", ""),
                "venue":        g.get("venue", {}).get("name", ""),
                "away_team":    away.get("team", {}).get("name", ""),
                "away_abbrev":  TEAM_ABBREV.get(away.get("team", {}).get("name", ""), "???"),
                "home_team":    home.get("team", {}).get("name", ""),
                "home_abbrev":  TEAM_ABBREV.get(home.get("team", {}).get("name", ""), "???"),
                "away_pitcher": _extract_pitcher(away),
                "home_pitcher": _extract_pitcher(home),
                "away_score":   away.get("score"),
                "home_score":   home.get("score"),
                "inning":       _extract_inning(g),
                "inning_state": _extract_inning_state(g),
            }
            games.append(game)

    _cache_set(cache_key, games)
    return games


def _extract_pitcher(team_data: dict) -> dict:
    pp = team_data.get("probablePitcher", {})
    return {
        "id":   pp.get("id"),
        "name": pp.get("fullName", "TBD"),
    }


def _extract_inning(g: dict) -> Optional[int]:
    ls = g.get("linescore", {})
    return ls.get("currentInning")


def _extract_inning_state(g: dict) -> str:
    ls = g.get("linescore", {})
    return ls.get("inningHalf", "")


async def get_live_game(game_id: int) -> dict:
    """Get live game feed with detailed play-by-play."""
    data = await _fetch_json(f"{MLB_STATS_BASE}.1/game/{game_id}/feed/live")
    if not data:
        return {}

    gd = data.get("gameData", {})
    ld = data.get("liveData", {})
    ls = ld.get("linescore", {})
    boxscore = ld.get("boxscore", {})

    return {
        "game_id":     game_id,
        "status":      gd.get("status", {}).get("detailedState", ""),
        "venue":       gd.get("venue", {}).get("name", ""),
        "away_team":   gd.get("teams", {}).get("away", {}).get("name", ""),
        "home_team":   gd.get("teams", {}).get("home", {}).get("name", ""),
        "away_score":  ls.get("teams", {}).get("away", {}).get("runs", 0),
        "home_score":  ls.get("teams", {}).get("home", {}).get("runs", 0),
        "inning":      ls.get("currentInning", 0),
        "inning_half": ls.get("inningHalf", ""),
        "outs":        ls.get("outs", 0),
        "balls":       ls.get("balls", 0),
        "strikes":     ls.get("strikes", 0),
        "total_runs":  (ls.get("teams", {}).get("away", {}).get("runs", 0) +
                        ls.get("teams", {}).get("home", {}).get("runs", 0)),
        "linescore":   ls,
        "plays":       ld.get("plays", {}),
    }


async def get_team_stats(team_id: int, season: int = None) -> dict:
    """Get team season stats."""
    season = season or datetime.now().year
    cache_key = f"team_stats_{team_id}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/teams/{team_id}/stats",
        {"stats": "season", "season": season, "group": "hitting,pitching"}
    )

    result = {"hitting": {}, "pitching": {}}
    for stat_group in data.get("stats", []):
        group_name = stat_group.get("group", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if splits:
            result[group_name] = splits[0].get("stat", {})

    _cache_set(cache_key, result)
    return result


async def get_player_stats(player_id: int, season: int = None) -> dict:
    """Get individual player season stats."""
    season = season or datetime.now().year
    cache_key = f"player_stats_{player_id}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/people/{player_id}",
        {"hydrate": f"stats(group=[hitting,pitching],type=[season],season={season})"}
    )

    people = data.get("people", [])
    if not people:
        return {}

    player = people[0]
    result = {
        "id":       player.get("id"),
        "name":     player.get("fullName", ""),
        "position": player.get("primaryPosition", {}).get("abbreviation", ""),
        "stats":    {},
    }

    for stat_group in player.get("stats", []):
        group_name = stat_group.get("group", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if splits:
            result["stats"][group_name] = splits[0].get("stat", {})

    _cache_set(cache_key, result)
    return result


async def get_pitcher_detailed(player_id: int, season: int = None) -> dict:
    """Get detailed pitcher stats including advanced metrics."""
    season = season or datetime.now().year
    cache_key = f"pitcher_detail_{player_id}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/people/{player_id}",
        {"hydrate": f"stats(group=[pitching],type=[season,career],season={season})"}
    )

    people = data.get("people", [])
    if not people:
        return {}

    player = people[0]
    result = {
        "id":      player.get("id"),
        "name":    player.get("fullName", ""),
        "season":  {},
        "career":  {},
    }

    for stat_group in player.get("stats", []):
        stat_type = stat_group.get("type", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if splits and stat_type in ("season", "career"):
            result[stat_type] = splits[0].get("stat", {})

    _cache_set(cache_key, result)
    return result


async def get_team_roster(team_id: int) -> list[dict]:
    """Get active roster for a team."""
    data = await _fetch_json(
        f"{MLB_STATS_BASE}/teams/{team_id}/roster",
        {"rosterType": "active"}
    )
    roster = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        roster.append({
            "id":       person.get("id"),
            "name":     person.get("fullName", ""),
            "position": entry.get("position", {}).get("abbreviation", ""),
            "status":   entry.get("status", {}).get("description", ""),
        })
    return roster


async def search_team_id(team_name: str) -> Optional[int]:
    """Look up a team's numeric ID from their name."""
    data = await _fetch_json(f"{MLB_STATS_BASE}/teams", {"sportId": 1})
    for team in data.get("teams", []):
        full = team.get("name", "").lower()
        abbr = team.get("abbreviation", "").lower()
        if team_name.lower() in full or team_name.lower() == abbr:
            return team.get("id")
    return None


async def get_standings() -> list[dict]:
    """Get current MLB standings."""
    season = datetime.now().year
    data = await _fetch_json(
        f"{MLB_STATS_BASE}/standings",
        {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
    )
    standings = []
    for record in data.get("records", []):
        div = record.get("division", {}).get("name", "")
        for team_rec in record.get("teamRecords", []):
            standings.append({
                "division":    div,
                "team":        team_rec.get("team", {}).get("name", ""),
                "wins":        team_rec.get("wins", 0),
                "losses":      team_rec.get("losses", 0),
                "pct":         team_rec.get("winningPercentage", ".000"),
                "gb":          team_rec.get("gamesBack", "-"),
                "streak":      team_rec.get("streak", {}).get("streakCode", ""),
                "last10":      f'{team_rec.get("records", {}).get("splitRecords", [{}])[0].get("wins", 0)}-{team_rec.get("records", {}).get("splitRecords", [{}])[0].get("losses", 0)}',
                "run_diff":    team_rec.get("runDifferential", 0),
            })
    return standings


# ═══════════════════════════════════════════════════════════════════════════
#  BASEBALL SAVANT / STATCAST
# ═══════════════════════════════════════════════════════════════════════════

async def get_pitcher_statcast(player_id: int, season: int = None) -> dict:
    """Get Statcast pitcher metrics from Baseball Savant."""
    season = season or datetime.now().year
    cache_key = f"savant_pitcher_{player_id}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    url = f"{SAVANT_BASE}/player-services/statcast-pitching-breakdown"
    text = await _fetch_text(url, {
        "playerId": player_id,
        "season": season,
        "position": "pitcher",
    })

    # Fallback: try the leaderboard CSV
    if not text or len(text) < 50:
        csv_url = f"{SAVANT_BASE}/leaderboard/custom"
        text = await _fetch_text(csv_url, {
            "year": season,
            "type": "pitcher",
            "min": 1,
            "csv": "true",
        })

    result = {"player_id": player_id, "season": season, "available": False}

    if text and len(text) > 100:
        try:
            df = pd.read_csv(StringIO(text))
            if "player_id" in df.columns:
                row = df[df["player_id"] == player_id]
                if not row.empty:
                    result.update(row.iloc[0].to_dict())
                    result["available"] = True
        except Exception as e:
            log.warning(f"Statcast parse error: {e}")

    _cache_set(cache_key, result)
    return result


async def get_batter_statcast(player_id: int, season: int = None) -> dict:
    """Get Statcast batter metrics from Baseball Savant."""
    season = season or datetime.now().year
    cache_key = f"savant_batter_{player_id}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    csv_url = f"{SAVANT_BASE}/leaderboard/custom"
    text = await _fetch_text(csv_url, {
        "year": season,
        "type": "batter",
        "min": 1,
        "csv": "true",
    })

    result = {"player_id": player_id, "season": season, "available": False}

    if text and len(text) > 100:
        try:
            df = pd.read_csv(StringIO(text))
            if "player_id" in df.columns:
                row = df[df["player_id"] == player_id]
                if not row.empty:
                    result.update(row.iloc[0].to_dict())
                    result["available"] = True
        except Exception as e:
            log.warning(f"Statcast parse error: {e}")

    _cache_set(cache_key, result)
    return result


async def get_matchup_data(pitcher_id: int, batter_ids: list[int],
                           season: int = None) -> list[dict]:
    """Get pitcher vs batter matchup history from Statcast."""
    season = season or datetime.now().year
    cache_key = f"matchup_{pitcher_id}_{'_'.join(map(str, batter_ids[:5]))}_{season}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # Use statcast search for pitcher vs batter
    results = []
    for batter_id in batter_ids[:9]:  # limit to avoid rate limiting
        text = await _fetch_text(
            f"{SAVANT_BASE}/statcast_search/csv",
            {
                "hfPT": "",
                "hfAB": "",
                "hfGT": "R|",
                "hfPR": "",
                "hfZ": "",
                "stadium": "",
                "hfBBL": "",
                "hfNewZones": "",
                "hfPull": "",
                "hfC": "",
                "hfSit": "",
                "hfOuts": "",
                "hfOpponent": "",
                "hfSA": "",
                "player_type": "pitcher",
                "hfInfield": "",
                "hfOutfield": "",
                "hfInn": "",
                "hfBBT": "",
                "hfFlag": "",
                "metric_1": "",
                "group_by": "name",
                "min_pitches": 0,
                "min_results": 0,
                "min_pas": 0,
                "sort_col": "pitches",
                "player_event_sort": "api_p_release_speed",
                "sort_order": "desc",
                "pitchers_lookup[]": pitcher_id,
                "batters_lookup[]": batter_id,
                "type": "details",
            }
        )
        if text and len(text) > 100:
            try:
                df = pd.read_csv(StringIO(text))
                if not df.empty:
                    ab_results = df.get("events", pd.Series()).dropna()
                    results.append({
                        "batter_id": batter_id,
                        "total_pa":  len(df),
                        "hits":      len(ab_results[ab_results.isin(["single", "double", "triple", "home_run"])]),
                        "k":         len(ab_results[ab_results == "strikeout"]),
                        "bb":        len(ab_results[ab_results == "walk"]),
                        "hr":        len(ab_results[ab_results == "home_run"]),
                        "avg_ev":    df.get("launch_speed", pd.Series()).mean(),
                    })
            except Exception:
                pass
        await asyncio.sleep(0.3)  # be polite

    _cache_set(cache_key, results)
    return results


async def get_team_recent_form(team_name: str, n_games: int = 14) -> dict:
    """Get a team's recent form over last N games."""
    team_id = await search_team_id(team_name)
    if not team_id:
        return {"team": team_name, "available": False}

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": "R",
            "hydrate": "linescore",
        }
    )

    results = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            is_home = home.get("team", {}).get("id") == team_id
            team_data = home if is_home else away
            opp_data = away if is_home else home
            results.append({
                "date":       g.get("officialDate", ""),
                "is_home":    is_home,
                "runs_for":   team_data.get("score", 0),
                "runs_against": opp_data.get("score", 0),
                "won":        team_data.get("isWinner", False),
            })

    results = results[-n_games:]
    wins = sum(1 for r in results if r["won"])
    total = len(results)
    avg_rf = sum(r["runs_for"] for r in results) / max(total, 1)
    avg_ra = sum(r["runs_against"] for r in results) / max(total, 1)

    return {
        "team":         team_name,
        "games":        total,
        "wins":         wins,
        "losses":       total - wins,
        "win_pct":      wins / max(total, 1),
        "avg_runs_for": round(avg_rf, 2),
        "avg_runs_ag":  round(avg_ra, 2),
        "run_diff":     round(avg_rf - avg_ra, 2),
        "available":    total > 0,
        "results":      results,
    }


async def get_head_to_head(team1: str, team2: str, season: int = None) -> dict:
    """Get head-to-head record for two teams this season."""
    season = season or datetime.now().year
    team1_id = await search_team_id(team1)
    team2_id = await search_team_id(team2)

    if not team1_id or not team2_id:
        return {"available": False}

    data = await _fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        {
            "sportId": 1,
            "teamId": team1_id,
            "startDate": f"{season}-01-01",
            "endDate": f"{season}-12-31",
            "gameType": "R",
            "hydrate": "linescore",
        }
    )

    matchups = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            away_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
            home_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
            if set([away_id, home_id]) == set([team1_id, team2_id]):
                if g.get("status", {}).get("detailedState") == "Final":
                    away = g.get("teams", {}).get("away", {})
                    home = g.get("teams", {}).get("home", {})
                    matchups.append({
                        "date":       g.get("officialDate", ""),
                        "away_team":  away.get("team", {}).get("name", ""),
                        "home_team":  home.get("team", {}).get("name", ""),
                        "away_score": away.get("score", 0),
                        "home_score": home.get("score", 0),
                    })

    team1_wins = sum(
        1 for m in matchups
        if (m["home_team"] == team1 and m["home_score"] > m["away_score"])
        or (m["away_team"] == team1 and m["away_score"] > m["home_score"])
    )

    return {
        "team1":      team1,
        "team2":      team2,
        "games":      len(matchups),
        "team1_wins": team1_wins,
        "team2_wins": len(matchups) - team1_wins,
        "matchups":   matchups,
        "available":  len(matchups) > 0,
    }
