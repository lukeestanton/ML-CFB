# clients/cfbd_client.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import requests

def _camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _convert_keys_to_snake_case(data: Dict[str, Any]) -> Dict[str, Any]:
    return {_camel_to_snake(k): v for k, v in data.items()}

class Game:
    def __init__(self, **kwargs):
        known_fields = {
            'id', 'season', 'week', 'season_type', 'start_date',
            'neutral_site', 'conference_game', 'home_team', 'away_team',
            'home_points', 'away_points', 'venue'
        }
        for field in known_fields:
            setattr(self, field, kwargs.get(field))
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class CFBDClient:
    BASE_URL = "https://api.collegefootballdata.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        })
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

class GamesAPI:
    def __init__(self, client: CFBDClient):
        self.client = client
    
    def get_games(
        self,
        year: Optional[int] = None,
        week: Optional[int] = None,
        season_type: Optional[str] = None,
    ) -> List[Game]:
        params = {}
        if year is not None:
            params["year"] = year
        if week is not None:
            params["week"] = week
        if season_type is not None:
            params["seasonType"] = season_type
        
        data = self.client._get("/games", params=params)
        return [Game(**_convert_keys_to_snake_case(item)) for item in data]

class BettingAPI:
    def __init__(self, client: CFBDClient):
        self.client = client
    
    def get_lines(
        self,
        year: Optional[int] = None,
        week: Optional[int] = None,
        season_type: Optional[str] = None,
    ) -> List[Dict]:
        params = {}
        if year is not None:
            params["year"] = year
        if week is not None:
            params["week"] = week
        if season_type is not None:
            params["seasonType"] = season_type
        
        return self.client._get("/lines", params=params)

class StatsAPI:
    def __init__(self, client: CFBDClient):
        self.client = client

    def get_advanced_game_stats(
        self,
        year: Optional[int] = None,
        week: Optional[int] = None,
        season_type: Optional[str] = "regular",
        exclude_garbage_time: Optional[bool] = False,
    ) -> List[Dict]:
        params = {}
        if year is not None:
            params["year"] = year
        if week is not None:
            params["week"] = week
        if season_type is not None:
            params["seasonType"] = season_type
        if exclude_garbage_time is not None:
            params["excludeGarbageTime"] = str(exclude_garbage_time).lower()

        return self.client._get("/stats/game/advanced", params=params)

def build_cfbd_client(api_key: str) -> CFBDClient:
    return CFBDClient(api_key)

def get_games_api(client: CFBDClient) -> GamesAPI:
    return GamesAPI(client)

def get_betting_api(client: CFBDClient) -> BettingAPI:
    return BettingAPI(client)

def get_stats_api(client: CFBDClient) -> StatsAPI:
    return StatsAPI(client)