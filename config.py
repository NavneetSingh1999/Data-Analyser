"""Credentials and settings. Nothing is hard-coded; the app runs without any of this.

To link your Shoonya API later, copy .env.example to .env and fill it in.
Real environment variables always win over the .env file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Automatically ensure categorical sub-folders are in sys.path
for _subdir in ("analytics", "data", "execution", "assets"):
    _sub_p = str(ROOT / _subdir)
    if _sub_p not in sys.path:
        sys.path.insert(0, _sub_p)

DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
ENV_FILE = ROOT / ".env"
WATCHLIST_FILE = DATA_DIR / "watchlist.txt" if (DATA_DIR / "watchlist.txt").exists() else (DATA_DIR / "watchlist.txt" if DATA_DIR != ROOT else ROOT / "watchlist.txt")
WATCHLISTS_JSON_FILE = DATA_DIR / "watchlists.json" if (DATA_DIR / "watchlists.json").exists() else (DATA_DIR / "watchlists.json" if DATA_DIR != ROOT else ROOT / "watchlists.json")
DEFAULT_WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"]

DEFAULT_WATCHLISTS: dict[str, list[str]] = {
    "My Watchlist": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"],
    "NIFTY Heavyweights": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "ITC", "LT", "SBIN", "KOTAKBANK"],
    "Banking Giants": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BANKBARODA", "INDUSINDBK"],
    "IT Titans": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE"],
    "Auto Champions": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR"],
}

# credential key -> environment variable
SHOONYA_FIELDS = {
    "user": "SHOONYA_USER",
    "password": "SHOONYA_PASSWORD",
    "totp_secret": "SHOONYA_TOTP_SECRET",
    "vendor_code": "SHOONYA_VENDOR_CODE",
    "api_key": "SHOONYA_API_KEY",
    "imei": "SHOONYA_IMEI",
}


def load_env(path: Path = ENV_FILE) -> None:
    """Minimal .env reader (KEY=VALUE lines). Does not override existing variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def shoonya_credentials() -> dict:
    load_env()
    return {k: os.environ.get(env, "").strip() for k, env in SHOONYA_FIELDS.items()}


def shoonya_missing(creds: dict | None = None) -> list[str]:
    """Names of the environment variables that are still empty."""
    creds = creds if creds is not None else shoonya_credentials()
    return [SHOONYA_FIELDS[k] for k in SHOONYA_FIELDS if not creds.get(k)]


def default_source() -> str:
    """'shoonya' once every credential is configured, otherwise 'sample'."""
    return "sample" if shoonya_missing() else "shoonya"


def load_watchlist(path: Path = WATCHLIST_FILE) -> list[str]:
    if not path.exists():
        return list(DEFAULT_WATCHLIST)
    syms = [s.strip().upper() for s in path.read_text(encoding="utf-8").splitlines()]
    return [s for s in syms if s and not s.startswith("#")] or list(DEFAULT_WATCHLIST)


def save_watchlist(symbols: list[str], path: Path = WATCHLIST_FILE) -> None:
    path.write_text("\n".join(dict.fromkeys(s.strip().upper() for s in symbols if s.strip())) + "\n", encoding="utf-8")


def load_all_watchlists(path: Path = WATCHLISTS_JSON_FILE) -> dict[str, list[str]]:
    """Load all named watchlists from watchlists.json. Falls back to watchlist.txt or presets."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    # Seed from existing watchlist.txt
    base_syms = load_watchlist()
    initial = dict(DEFAULT_WATCHLISTS)
    initial["My Watchlist"] = base_syms
    try:
        path.write_text(json.dumps(initial, indent=2), encoding="utf-8")
    except Exception:
        pass
    return initial


def save_all_watchlists(wls: dict[str, list[str]], path: Path = WATCHLISTS_JSON_FILE) -> None:
    """Save all named watchlists to watchlists.json and sync primary watchlist to watchlist.txt."""
    try:
        path.write_text(json.dumps(wls, indent=2), encoding="utf-8")
    except Exception:
        pass
    if wls:
        first_key = "My Watchlist" if "My Watchlist" in wls else next(iter(wls.keys()))
        save_watchlist(wls[first_key])


def create_watchlist(name: str, symbols: list[str] | None = None) -> bool:
    """Create a new named watchlist. Returns True on success, False if name exists or invalid."""
    clean_name = name.strip()
    if not clean_name:
        return False
    wls = load_all_watchlists()
    if clean_name in wls:
        return False
    wls[clean_name] = [s.strip().upper() for s in (symbols or []) if s.strip()]
    save_all_watchlists(wls)
    return True


def delete_watchlist(name: str) -> bool:
    """Delete a named watchlist. Returns True if deleted (prevents deleting last remaining)."""
    clean_name = name.strip()
    wls = load_all_watchlists()
    if clean_name not in wls or len(wls) <= 1:
        return False
    del wls[clean_name]
    save_all_watchlists(wls)
    return True


def add_to_watchlist(name: str, symbol: str) -> bool:
    """Add a symbol to a named watchlist. Returns True if newly added."""
    clean_name = name.strip()
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        return False
    wls = load_all_watchlists()
    if clean_name not in wls:
        wls[clean_name] = []
    if clean_sym not in wls[clean_name]:
        wls[clean_name].append(clean_sym)
        save_all_watchlists(wls)
        return True
    return False


def remove_from_watchlist(name: str, symbol: str) -> bool:
    """Remove a symbol from a named watchlist. Returns True if removed."""
    clean_name = name.strip()
    clean_sym = symbol.strip().upper()
    wls = load_all_watchlists()
    if clean_name in wls and clean_sym in wls[clean_name]:
        wls[clean_name].remove(clean_sym)
        save_all_watchlists(wls)
        return True
    return False


def rename_watchlist(old_name: str, new_name: str) -> bool:
    """Rename a watchlist from old_name to new_name."""
    o_clean = old_name.strip()
    n_clean = new_name.strip()
    if not o_clean or not n_clean or o_clean == n_clean:
        return False
    wls = load_all_watchlists()
    if o_clean not in wls or n_clean in wls:
        return False
    wls[n_clean] = wls.pop(o_clean)
    save_all_watchlists(wls)
    return True
