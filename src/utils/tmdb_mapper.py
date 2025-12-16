from typing import Any, Dict, Optional
from datetime import datetime

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def map_tmdb_details_to_model_input(details: Dict[str, Any]) -> Dict[str, Any]:
    release_date = details.get("release_date") or ""
    release_month: Optional[int] = None

    if release_date:
        try:
            release_month = datetime.strptime(release_date, "%Y-%m-%d").month
        except Exception:
            release_month = None

    return {
        "budget": _safe_float(details.get("budget"), 0.0),
        "popularity": _safe_float(details.get("popularity"), 0.0),
        "runtime": _safe_float(details.get("runtime"), 0.0),
        "vote_average": _safe_float(details.get("vote_average"), 0.0),
        "vote_count": _safe_float(details.get("vote_count"), 0.0),
        "release_month": int(release_month) if release_month else 1,

        # Optional extra metadata for UI/debug (model can ignore)
        "tmdb_id": details.get("id"),
        "title": details.get("title"),
        "release_date": details.get("release_date"),
    }
