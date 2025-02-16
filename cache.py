import json
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_file_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"


def get_cached_summary(file_hash: str) -> dict:
    cache_file = get_cache_file_path(file_hash)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def set_cached_summary(file_hash: str, summary_data: dict) -> None:
    cache_file = get_cache_file_path(file_hash)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
