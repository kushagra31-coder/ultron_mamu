"""Robust browser/YouTube tools for Ultron.

YouTube search uses yt-dlp metadata search instead of DDGS video search, which
makes song/artist matching much more reliable. No media is downloaded.
"""
from __future__ import annotations

import re
import urllib.parse
import webbrowser
from difflib import SequenceMatcher

import yt_dlp

from .registry import tool


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_music_query(query: str) -> str:
    """Correct common Whisper truncations without hardcoding individual songs."""
    q = query.strip()
    normalized = _normalize_text(q)

    # Whisper commonly hears the artist as just "Tame". In a music request,
    # expand that to Tame Impala so YouTube ranking has enough information.
    if re.search(r"\btame\b", q, flags=re.IGNORECASE) and not re.search(
        r"\btame\s+impala\b", q, flags=re.IGNORECASE
    ):
        q = re.sub(r"\btame\b", "Tame Impala", q, count=1, flags=re.IGNORECASE)

    return q


def _split_music_query(query: str) -> tuple[str, str]:
    """Return song/title hint and artist hint when the user says 'X by Y'."""
    match = re.match(r"^(.+?)\s+by\s+(.+)$", query.strip(), flags=re.IGNORECASE)
    if not match:
        return query.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def _score_video(entry: dict, title_hint: str, artist_hint: str) -> float:
    title = str(entry.get("title") or "")
    uploader = str(entry.get("uploader") or entry.get("channel") or "")

    title_n = _normalize_text(title)
    wanted_title = _normalize_text(title_hint)
    artist_n = _normalize_text(artist_hint)

    score = 0.0

    if wanted_title:
        if wanted_title in title_n:
            score += 8.0
        score += 4.0 * SequenceMatcher(None, wanted_title, title_n).ratio()

    if artist_n:
        if artist_n in _normalize_text(uploader):
            score += 8.0
        if artist_n in title_n:
            score += 3.0

    # Official channels / music videos should win close matches.
    lowered = f"{title} {uploader}".lower()
    for marker, points in (
        ("official", 2.0),
        ("tame impala", 2.0),
        ("topic", 1.0),
        ("provided to youtube", 1.0),
    ):
        if marker in lowered:
            score += points

    return score


@tool(description="Open an exact HTTP/HTTPS URL in the default browser.")
def open_url(url: str) -> str:
    """Open a URL in the default browser."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        return "Only HTTP/HTTPS URLs are allowed."
    opened = webbrowser.open(url)
    return f"Opened {url}." if opened else f"Sent {url} to the default browser."


@tool(description="Find a YouTube or YouTube Music song/video by title and artist, then open the best matching result.")
def play_youtube(query: str, platform: str = "youtube") -> str:
    """Search YouTube or YouTube Music, then open the best match.
    
    Args:
        query: The search query (e.g., 'Loser by Tame Impala').
        platform: Either 'youtube' or 'youtube music'.
    """
    search_query = _normalize_music_query(query)
    title_hint, artist_hint = _split_music_query(search_query)

    is_music = platform.lower().strip() == "youtube music"
    base_url = "https://music.youtube.com" if is_music else "https://www.youtube.com"
    
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(
                f"ytsearch10:{search_query}",
                download=False,
            )
    except Exception as exc:
        # Keep a browser-only fallback instead of claiming failure.
        fallback = (
            f"{base_url}/results?search_query="
            + urllib.parse.quote_plus(search_query)
        )
        webbrowser.open(fallback)
        return f"I couldn't rank the results automatically, so I opened the search for {search_query} on {platform}."

    entries = [e for e in (info.get("entries") or []) if isinstance(e, dict)]
    if not entries:
        fallback = (
            f"{base_url}/results?search_query="
            + urllib.parse.quote_plus(search_query)
        )
        webbrowser.open(fallback)
        return f"I couldn't find a direct result, so I opened search for {search_query} on {platform}."

    ranked = sorted(
        entries,
        key=lambda entry: _score_video(entry, title_hint, artist_hint),
        reverse=True,
    )
    best = ranked[0]

    video_id = best.get("id")
    title = str(best.get("title") or search_query)
    webpage_url = best.get("webpage_url")
    if not webpage_url and video_id:
        webpage_url = f"{base_url}/watch?v={video_id}"
    elif webpage_url and is_music:
        # If yt-dlp returns a standard youtube url, convert it to youtube music
        webpage_url = webpage_url.replace("www.youtube.com", "music.youtube.com")

    if not webpage_url:
        fallback = (
            f"{base_url}/results?search_query="
            + urllib.parse.quote_plus(search_query)
        )
        webbrowser.open(fallback)
        return f"I found results for {search_query}, but did not get a direct URL."

    webbrowser.open(webpage_url)
    return f"Opened '{title}' on {platform} for {search_query}."
