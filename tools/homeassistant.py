"""Home Assistant integration — free, local smart-home control.

Configure with environment variables:
    ULTRON_HA_URL    Home Assistant base URL (default http://homeassistant.local:8123)
    ULTRON_HA_TOKEN  Long-lived access token (HA → Profile → Long-lived access tokens)

All tools degrade gracefully when the token is missing: they explain what
to set instead of failing cryptically.
"""
from __future__ import annotations

import json
import os

import requests

from .registry import tool

HA_URL = os.getenv("ULTRON_HA_URL", "http://homeassistant.local:8123").rstrip("/")
HA_TOKEN = os.getenv("ULTRON_HA_TOKEN", "")
_TIMEOUT = 10


def _ha():
    """Return (base_url, headers) or an error string when unconfigured."""
    if not HA_TOKEN:
        return (
            "Home Assistant is not configured. Set ULTRON_HA_URL (default "
            "http://homeassistant.local:8123) and ULTRON_HA_TOKEN (a long-lived "
            "access token from Home Assistant → your Profile → "
            "'Long-lived access tokens'), then restart the agent."
        )
    return HA_URL, {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _friendly(state: dict) -> str:
    attrs = state.get("attributes", {}) or {}
    name = attrs.get("friendly_name", state.get("entity_id"))
    return f"{name} ({state.get('entity_id')}): {state.get('state')}"


@tool(
    description=(
        "Get the current state of a Home Assistant entity, e.g. "
        "'light.living_room' or 'sensor.temperature'. Returns its state "
        "and friendly name."
    )
)
def ha_get_state(entity_id: str) -> str:
    """Read a Home Assistant entity's state."""
    cfg = _ha()
    if isinstance(cfg, str):
        return cfg
    base, headers = cfg
    entity_id = entity_id.strip()
    if not entity_id:
        return "No entity_id was provided."
    try:
        resp = requests.get(f"{base}/api/states/{entity_id}", headers=headers, timeout=_TIMEOUT)
    except Exception as exc:
        return f"Could not reach Home Assistant at {base}: {exc}"
    if resp.status_code == 404:
        return f"Entity '{entity_id}' was not found in Home Assistant."
    if resp.status_code != 200:
        return f"Home Assistant returned HTTP {resp.status_code}: {resp.text[:200]}"
    return _friendly(resp.json())


@tool(
    description=(
        "Call a Home Assistant service, e.g. domain='light', service='toggle', "
        "entity_id='light.living_room'. service_data is optional JSON for "
        "extra fields like brightness (e.g. '{\"brightness\": 128}')."
    )
)
def ha_call_service(domain: str, service: str, entity_id: str = "", service_data: str = "{}") -> str:
    """Call a Home Assistant service on an entity."""
    cfg = _ha()
    if isinstance(cfg, str):
        return cfg
    base, headers = cfg
    domain, service = domain.strip(), service.strip()
    if not domain or not service:
        return "Both domain and service are required (e.g. domain='light', service='toggle')."
    try:
        data = json.loads(service_data or "{}")
    except json.JSONDecodeError:
        return "service_data must be valid JSON."
    if entity_id.strip():
        data["entity_id"] = entity_id.strip()
    try:
        resp = requests.post(
            f"{base}/api/services/{domain}/{service}",
            headers=headers, json=data, timeout=_TIMEOUT,
        )
    except Exception as exc:
        return f"Could not reach Home Assistant at {base}: {exc}"
    if resp.status_code not in (200, 201):
        return f"Home Assistant returned HTTP {resp.status_code}: {resp.text[:200]}"
    target = f" on {entity_id.strip()}" if entity_id.strip() else ""
    return f"Called {domain}.{service}{target}."


@tool(
    description=(
        "List Home Assistant entities, optionally filtered by domain "
        "(e.g. domain='light' shows only lights). Use this to discover "
        "entity names before controlling them."
    )
)
def ha_list_entities(domain: str = "") -> str:
    """List entities known to Home Assistant, optionally by domain."""
    cfg = _ha()
    if isinstance(cfg, str):
        return cfg
    base, headers = cfg
    domain = domain.strip().lower().rstrip(".") + ("." if domain.strip() else "")
    try:
        resp = requests.get(f"{base}/api/states", headers=headers, timeout=_TIMEOUT)
    except Exception as exc:
        return f"Could not reach Home Assistant at {base}: {exc}"
    if resp.status_code != 200:
        return f"Home Assistant returned HTTP {resp.status_code}: {resp.text[:200]}"
    states = resp.json()
    if domain:
        states = [s for s in states if s.get("entity_id", "").startswith(domain)]
    if not states:
        return f"No entities found for domain '{domain or 'all'}'."
    lines = [_friendly(s) for s in states[:40]]
    more = f" (+{len(states) - 40} more)" if len(states) > 40 else ""
    return "\n".join(lines) + more
