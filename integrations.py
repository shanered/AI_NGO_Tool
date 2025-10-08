# integrations.py
# Fetchers for external donor portals and data sources with simple normalization and verification flags.

from __future__ import annotations
import time
from typing import Dict, Any, List
import requests
import xml.etree.ElementTree as ET

DEFAULT_TIMEOUT = 20

ISO3_MAP = {
    "Somalia": "SOM", "Uganda": "UGA", "Ethiopia": "ETH", "Niger": "NER", "Liberia": "LBR",
    "Mozambique": "MOZ", "Bangladesh": "BGD", "Nepal": "NPL", "Sri Lanka": "LKA", "Myanmar": "MMR",
    "Indonesia": "IDN", "Kenya": "KEN", "Nigeria": "NGA", "Vietnam": "VNM", "Pakistan": "PAK",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ok(items: List[Dict[str, Any]]) -> bool:
    return bool(items)


def fetch_europeaid_rss(country: str) -> Dict[str, Any]:
    """Fetch EuropeAid/INTPA RSS feed and filter by country substring (best-effort).
    Source: https://international-partnerships.ec.europa.eu/rss_en
    """
    url = "https://international-partnerships.ec.europa.eu/rss_en"
    verified = False
    items: List[Dict[str, Any]] = []
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        verified = True  # Official EU domain
        # Parse RSS XML
        root = ET.fromstring(r.text)
        # Typical structure: rss/channel/item
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pubDate = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            if country.lower() in title.lower():
                items.append({"title": title, "url": link, "date": pubDate, "verified": True})
            else:
                # Keep a small sample even if not country-filtered
                if len(items) < 2:
                    items.append({"title": title, "url": link, "date": pubDate, "verified": True})
    except Exception as e:
        return {"source": "EuropeAid/INTPA", "country": country, "verified": False, "error": str(e), "items": [], "source_url": url, "fetched_at": _now()}
    return {"source": "EuropeAid/INTPA", "country": country, "verified": verified and _ok(items), "items": items, "source_url": url, "fetched_at": _now()}


def fetch_fdco_projects(country: str) -> Dict[str, Any]:
    """Fetch FCDO DevTracker projects by country.
    Attempt ISO3 then ISO2; returns small sample. Verified if HTTP 200 with items.
    Example (may vary): https://devtracker.fcdo.gov.uk/api/countries/UGA/projects?limit=5
    """
    base = "https://devtracker.fcdo.gov.uk/api/countries/{code}/projects"
    iso3 = ISO3_MAP.get(country)
    params = {"limit": 5}
    for code in filter(None, [iso3, iso3[:2] if iso3 else None]):
        try:
            url = base.format(code=code)
            r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                js = r.json()
                items = []
                # Expect a list of projects
                for obj in (js if isinstance(js, list) else [] )[:5]:
                    title = obj.get("title") or obj.get("project_title") or "Project"
                    uid = obj.get("iati_identifier") or obj.get("id") or ""
                    link = f"https://devtracker.fcdo.gov.uk/projects/{uid}" if uid else ""
                    items.append({"title": title, "url": link, "verified": True})
                return {"source": "FCDO DevTracker", "country": country, "verified": bool(items), "items": items, "source_url": url, "fetched_at": _now()}
        except Exception:
            continue
    return {"source": "FCDO DevTracker", "country": country, "verified": False, "items": [], "source_url": base, "fetched_at": _now(), "error": "No API response or unrecognized country code"}


def fetch_usaid_rss(country: str) -> Dict[str, Any]:
    """Fetch USAID press releases RSS and filter by country keyword.
    RSS: https://www.usaid.gov/news-information/press-releases?format=xml
    """
    url = "https://www.usaid.gov/news-information/press-releases?format=xml"
    items: List[Dict[str, Any]] = []
    verified = False
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        verified = True
        root = ET.fromstring(r.text)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pubDate = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            if country.lower() in title.lower():
                items.append({"title": title, "url": link, "date": pubDate, "verified": True})
        # if none matched, keep a small sample
        if not items:
            for item in root.findall(".//item")[:3]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pubDate = (item.findtext("pubDate") or "").strip()
                if title and link:
                    items.append({"title": title, "url": link, "date": pubDate, "verified": True})
    except Exception as e:
        return {"source": "USAID RSS", "country": country, "verified": False, "items": [], "source_url": url, "fetched_at": _now(), "error": str(e)}
    return {"source": "USAID RSS", "country": country, "verified": _ok(items), "items": items, "source_url": url, "fetched_at": _now()}


def fetch_bmgf_countries() -> Dict[str, Any]:
    """Attempt to fetch Gates Foundation 'Where we work'. If blocked, mark UNVERIFIED with reason.
    A simple GET to main page; if we cannot parse structured countries, return unverified.
    """
    url = "https://www.gatesfoundation.org/our-work"
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        # Page is dynamically rendered; without JS we cannot parse robustly. Mark partial.
        return {"source": "BMGF", "verified": False, "items": [], "source_url": url, "fetched_at": _now(), "note": "Dynamic site; needs JS rendering for country list"}
    except Exception as e:
        return {"source": "BMGF", "verified": False, "items": [], "source_url": url, "fetched_at": _now(), "error": str(e)}


def fetch_aiddata(country: str) -> Dict[str, Any]:
    """Attempt AidData country search (HTML fallback). Mark UNVERIFIED if JSON not available.
    """
    url = f"https://www.aiddata.org/search?q={requests.utils.quote(country)}"
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        # HTML page; without dataset API, mark as UNVERIFIED content-only
        return {"source": "AidData", "country": country, "verified": False, "items": [], "source_url": url, "fetched_at": _now(), "note": "No public JSON; HTML search fetched"}
    except Exception as e:
        return {"source": "AidData", "country": country, "verified": False, "items": [], "source_url": url, "fetched_at": _now(), "error": str(e)}


# -------- World Bank ODA (DT.ODA.ODAT.CD) --------

def fetch_worldbank_oda(iso3: str, years: int = 5) -> List[Dict[str, Any]]:
    """Fetch Net ODA received (current US$) for a country from World Bank API.

    Endpoint:
    https://api.worldbank.org/v2/country/{ISO3}/indicator/DT.ODA.ODAT.CD?format=json

    Returns a list of dicts sorted from most recent, limited to `years` entries.
    """
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/DT.ODA.ODAT.CD?format=json&per_page=100"
    r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    data = js[1] if isinstance(js, list) and len(js) > 1 else []
    series = [
        {"year": d.get("date"), "value": d.get("value")}
        for d in data if d and d.get("value") is not None
    ]
    # Sort by year desc and take last N
    try:
        series.sort(key=lambda x: int(x["year"]), reverse=True)
    except Exception:
        pass
    return series[:years]
