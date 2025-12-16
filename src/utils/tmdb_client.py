import os
from typing import Any, Dict, List, Optional
import httpx

TMDB_BASE_URL = "https://api.themoviedb.org/3"

def _auth_headers() -> Dict[str, str]:
    token = os.getenv("TMDB_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TMDB_BEARER_TOKEN is not set")
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}

async def tmdb_search_movie(query: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    params = {"query": query}
    if year:
        params["year"] = year
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{TMDB_BASE_URL}/search/movie", headers=_auth_headers(), params=params)
        r.raise_for_status()
        return r.json().get("results", [])

async def tmdb_movie_details(movie_id: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{TMDB_BASE_URL}/movie/{movie_id}", headers=_auth_headers())
        r.raise_for_status()
        return r.json()
