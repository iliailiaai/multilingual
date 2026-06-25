import requests
import re
import time
from datetime import datetime, timedelta
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

RETRY_WAIT_SECONDS = 80
REQUEST_TIMEOUT_SECONDS = 30


def _request_json(api_key: str, endpoint: str, params: dict, retry: int = 5) -> dict:
    url = f"{SportsDBClient.BASE_URL}/{api_key}/{endpoint}"

    for attempt in range(1, retry + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429:
                print(
                    f"[SportsDBClient] 429 Too Many Requests for {response.url}; "
                    f"waiting {RETRY_WAIT_SECONDS}s ({attempt}/{retry})..."
                )
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            response.raise_for_status()
            data = response.json()
            message = data.get("Message")
            if message:
                print(f"[SportsDBClient] API message for {response.url}: {message}")
            return data
        except requests.exceptions.RequestException as e:
            if attempt >= retry:
                print(f"[SportsDBClient] error after {retry} attempts: {e}")
                return {}

            print(
                f"[SportsDBClient] error: {e}; waiting {RETRY_WAIT_SECONDS}s "
                f"({attempt}/{retry})..."
            )
            time.sleep(RETRY_WAIT_SECONDS)

    return {}


class SportsDBClient:
    BASE_URL = "https://www.thesportsdb.com/api/v1/json"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, endpoint: str, params: dict) -> dict:
        return _request_json(self.api_key, endpoint, params)

    def get_events_by_date(self, date: str, league: str, window: int = 1) -> list:
        """
        Fetch events for date ± window days (default 1 = yesterday, today, tomorrow).
        date: str in "YYYY-MM-DD"
        league: league name
        """
        base_date = datetime.strptime(date, "%Y-%m-%d").date()
        dates_to_check = [
            base_date + timedelta(days=offset)
            for offset in range(-window, window + 1)
        ]

        all_events = []
        for d in dates_to_check:
            endpoint = "eventsday.php"
            params = {"d": d.isoformat(), "l": league}
            data = self._request(endpoint, params)
            events = data.get("events") or []
            all_events.extend(events)

        # Deduplicate by event ID
        seen = set()
        unique_events = []
        for e in all_events:
            if e["idEvent"] not in seen:
                seen.add(e["idEvent"])
                unique_events.append(e)

        return unique_events
    
    def get_events_by_season(self, league_id: str, season: str) -> list:
        data = self._request("eventsseason.php", {"id": league_id, "s": season})
        return data.get("events") or []

    def _season_strings_for_range(self, start, end, sport: str) -> list[str]:
        if sport == "soccer":
            return [f"{year}-{year + 1}" for year in range(start.year - 1, end.year + 1)]
        return [str(year) for year in range(start.year, end.year + 1)]

    def get_events_by_range(
        self,
        start_date: str,
        end_date: str,
        league: str,
        max_events: int,
        sport: str,
    ) -> list:
        """
        Fetch all events between start_date and end_date (inclusive).
        start_date, end_date: 'YYYY-MM-DD'
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        all_events = []

        for season in self._season_strings_for_range(start, end, sport):
            season_events = self.get_events_by_season(league, season)
            print(f"[SportsDBClient] got {len(season_events)} raw events for league {league}, season {season}")

            for event in season_events:
                date_str = event.get("dateEvent")
                if not date_str:
                    continue

                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                if start <= event_date <= end:
                    all_events.append(event)

            if len(all_events) >= max_events + 20:
                break

        # 去重複
        seen = set()
        unique_events = []
        for event in sorted(all_events, key=lambda e: e.get("dateEvent") or ""):
            event_id = event.get("idEvent")
            if event_id and event_id not in seen:
                seen.add(event_id)
                unique_events.append(event)

        return unique_events



# ===================== PARSERS =====================

def parse_baseball_result(result_str):
    teams_data = {}
    parts = result_str.split("<br><br>")
    for part in parts:
        lines = part.strip().split("<br>")
        if len(lines) < 3:
            continue
        team_name = lines[0].replace("Innings:", "").strip()
        innings_scores = [int(x) for x in lines[1].split() if x.isdigit()]
        hits_errors = re.findall(r"Hits:\s*(\d+)\s*-\s*Errors:\s*(\d+)", lines[2])
        hits, errors = (map(int, hits_errors[0]) if hits_errors else (None, None))
        teams_data[team_name] = {"innings": innings_scores, "hits": hits, "errors": errors}
    return teams_data


def get_baseball_event_details(api_key, event_id):
    data = _request_json(api_key, "lookupevent.php", {"id": event_id})
    events = data.get("events")
    if not events:
        return None
    event = events[0]
    parsed = parse_baseball_result(event.get("strResult", ""))
    return {
        "venue": event.get("strVenue"),
        "home_team": event.get("strHomeTeam"),
        "away_team": event.get("strAwayTeam"),
        "parsed_result": parsed,
    }


def get_soccer_event_stats(api_key, event_id):
    data = _request_json(api_key, "lookupeventstats.php", {"id": event_id})
    stats = data.get("eventstats")
    if not stats:
        return None
    result = {}
    for s in stats:
        stat_name = s["strStat"]
        result[stat_name] = {"home": s.get("intHome"), "away": s.get("intAway")}
    return result


if __name__ == "__main__":
    sports_client = SportsDBClient(api_key=os.getenv("SPORTSDB_API_KEY", ""))
    events = sports_client.get_events_by_range("2025-09-01", "2025-09-05", "4424", 5, "baseball")

    print(f"Total events: {len(events)}")
    print(events[:3])
    print(events[:-3])
    print(f"Total events: {len(events)}")
