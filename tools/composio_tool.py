"""Composio third-party app toolset (LibreChatHermes).

Bridges the agent to a user's connected third-party accounts — Google Drive,
Notion, Google Sheets, … — via Composio (https://composio.dev). Rather than
registering one agent tool per third-party action (which would require a
per-user, dynamic tool registry the gateway does not have), this exposes two
fixed **meta-tools** whose per-user behaviour is driven entirely by session
context — mirroring Composio's own ``COMPOSIO_SEARCH_TOOLS`` /
``COMPOSIO_*_EXECUTE`` pattern:

  * ``composio_search_tools``  — discover available actions for the user's
    allowed toolkits (returns slugs + input schemas).
  * ``composio_execute_tool``  — run one action as the user.

Per-user scoping (set per turn by ``gateway/platforms/api_server.py`` →
``set_session_vars``):

  * ``HERMES_SESSION_COMPOSIO_USER_ID`` — the LibreChatHermes DB user id, used
    directly as the Composio ``user_id`` so connected accounts stay isolated.
  * ``HERMES_SESSION_COMPOSIO_TOOLKITS`` — comma-joined toolkit allowlist
    (admin-granted ∩ user-connected) the agent may reach this turn.

The toolset is offered to a turn only when the server appends ``composio`` to
the agent's enabled toolsets (it does so iff a Composio user id is present); the
registry ``check_fn`` here only gates **global** availability (API key + SDK
present) and is intentionally session-independent because ``check_fn`` results
are process-cached. Execution additionally rejects any tool whose toolkit is not
in the session allowlist (defense in depth).
"""

import json
import logging
import os
from typing import Any, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# Per-turn result cap. Composio schemas/results can be large; keep them bounded
# so a single tool call cannot blow the model's context.
_MAX_RESULT_CHARS = 60_000

# Cache one Composio client per API key for the process lifetime.
_CLIENT_CACHE: dict = {}


def _api_key() -> str:
    return (os.getenv("COMPOSIO_API_KEY") or "").strip()


def _get_client():
    """Return a cached Composio client, or ``None`` when unavailable.

    Never raises — callers treat ``None`` as "Composio not configured".
    """
    key = _api_key()
    if not key:
        return None
    cached = _CLIENT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from composio import Composio  # lazy: keep import off the startup path
    except Exception as exc:  # pragma: no cover - import-time env issue
        logger.warning("composio: SDK import failed (%s); toolset disabled", exc)
        return None
    try:
        client = Composio(api_key=key)
    except Exception as exc:
        logger.warning("composio: client init failed (%s); toolset disabled", exc)
        return None
    _CLIENT_CACHE[key] = client
    return client


def _session_user_id() -> str:
    from gateway.session_context import get_session_env

    return (get_session_env("HERMES_SESSION_COMPOSIO_USER_ID") or "").strip()


def _allowed_toolkits() -> List[str]:
    """Lowercased toolkit slugs the user may reach this turn (may be empty)."""
    from gateway.session_context import get_session_env

    raw = get_session_env("HERMES_SESSION_COMPOSIO_TOOLKITS") or ""
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _toolkit_of_slug(slug: str) -> str:
    """Toolkit slug a Composio action belongs to.

    Composio action slugs are ``<TOOLKIT>_<ACTION>`` with the toolkit being the
    uppercased toolkit slug (e.g. ``GOOGLEDRIVE_DOWNLOAD_FILE`` → ``googledrive``).
    TODO: for toolkits whose slug itself contains an underscore this prefix split
    is approximate; the current catalog (googledrive/notion/googlesheets) is safe.
    """
    return slug.split("_", 1)[0].lower() if slug else ""


def check_composio_available() -> bool:
    """Global availability gate (session-independent — results are cached).

    True only when an API key is configured and the SDK imports. Per-user
    gating is handled separately by enabled-toolset membership.
    """
    return _get_client() is not None


def _err(message: str) -> str:
    return json.dumps({"error": message})


def _cap(text: str) -> str:
    if len(text) > _MAX_RESULT_CHARS:
        return text[: _MAX_RESULT_CHARS - 20] + '… (truncated)"}'
    return text


def _tool_summary(tool: Any) -> dict:
    """Compact, JSON-safe view of a raw Composio Tool: slug + description + inputs."""
    data: Optional[dict] = None
    dump = getattr(tool, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
        except Exception:
            data = None
    if not isinstance(data, dict):
        data = {
            "slug": getattr(tool, "slug", None) or getattr(tool, "name", None),
            "description": getattr(tool, "description", "") or "",
            "input_parameters": getattr(tool, "input_parameters", None) or {},
        }
    return {
        "slug": data.get("slug") or data.get("name"),
        "description": data.get("description", "") or "",
        "input_parameters": data.get("input_parameters")
        or data.get("inputParameters")
        or {},
    }


def _response_to_dict(resp: Any) -> dict:
    # composio.tools.execute returns a plain dict ({data, error, successful, ...}).
    if isinstance(resp, dict):
        return resp
    dump = getattr(resp, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return {
        "successful": getattr(resp, "successful", getattr(resp, "success", None)),
        "data": getattr(resp, "data", None),
        "error": getattr(resp, "error", None),
    }


# ── Handlers ────────────────────────────────────────────────────────────────


def composio_search_tools(args: dict, **_kw) -> str:
    client = _get_client()
    if client is None:
        return _err("Composio is not configured on this gateway (COMPOSIO_API_KEY unset).")
    user_id = _session_user_id()
    if not user_id:
        return _err("Composio is not enabled for this session.")

    allowed = _allowed_toolkits()
    if not allowed:
        return json.dumps(
            {"tools": [], "note": "No third-party apps are connected/enabled for you yet."}
        )

    query = str(args.get("query") or "").strip()
    requested = args.get("toolkits")
    if isinstance(requested, list) and requested:
        wanted = {str(t).strip().lower() for t in requested if str(t).strip()}
        toolkits = [t for t in allowed if t in wanted]
    else:
        toolkits = list(allowed)
    if not toolkits:
        return json.dumps(
            {"tools": [], "note": "None of the requested toolkits are enabled for you."}
        )

    limit = args.get("limit")
    try:
        limit = max(1, min(int(limit), 30)) if limit is not None else 10
    except (TypeError, ValueError):
        limit = 10

    try:
        raw = client.tools.get_raw_composio_tools(
            toolkits=[t.upper() for t in toolkits],
            search=query or None,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("composio: tool search failed: %s", exc)
        return _err(f"Composio tool search failed: {exc}")

    tools = [_tool_summary(t) for t in (raw or [])]
    return _cap(json.dumps({"tools": tools, "toolkits": toolkits}))


def composio_execute_tool(args: dict, **_kw) -> str:
    client = _get_client()
    if client is None:
        return _err("Composio is not configured on this gateway (COMPOSIO_API_KEY unset).")
    user_id = _session_user_id()
    if not user_id:
        return _err("Composio is not enabled for this session.")

    slug = str(args.get("slug") or "").strip()
    if not slug:
        return _err("Missing 'slug' (use composio_search_tools to find one).")

    allowed = _allowed_toolkits()
    toolkit = _toolkit_of_slug(slug)
    if toolkit not in allowed:
        return _err(
            f"Tool '{slug}' belongs to toolkit '{toolkit}', which is not enabled for you."
        )

    arguments = args.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _err("'arguments' must be an object.")

    try:
        # dangerously_skip_version_check: Composio otherwise requires a pinned
        # toolkit version per execute ("latest" isn't accepted for manual calls).
        # We run against the current toolkit version rather than pinning each one.
        resp = client.tools.execute(
            slug, arguments=arguments, user_id=user_id, dangerously_skip_version_check=True
        )
    except Exception as exc:
        logger.warning("composio: execute %s failed: %s", slug, exc)
        return _err(f"Composio execution failed: {exc}")

    return _cap(json.dumps(_response_to_dict(resp), default=str))


# ── Schemas ─────────────────────────────────────────────────────────────────

COMPOSIO_SEARCH_SCHEMA = {
    "name": "composio_search_tools",
    "description": (
        "Discover actions available on the user's connected third-party apps "
        "(e.g. Google Drive, Notion, Google Sheets) via Composio. Returns matching "
        "tool slugs and their input schemas. Call this first to find the exact slug "
        "and arguments, then call composio_execute_tool. Only the apps the user has "
        "connected and been granted are searchable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of the capability you need (e.g. 'list files in a folder', 'append a row').",
            },
            "toolkits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of toolkit slugs to search (e.g. ['googledrive']). Omit to search all of the user's enabled apps.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of tools to return (default 10, max 30).",
            },
        },
        "required": ["query"],
    },
}

COMPOSIO_EXECUTE_SCHEMA = {
    "name": "composio_execute_tool",
    "description": (
        "Execute one Composio action on behalf of the current user, using their "
        "connected third-party account. Pass the exact 'slug' from "
        "composio_search_tools and an 'arguments' object matching that tool's input "
        "schema. Runs as the user; you cannot act on apps they have not connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Exact Composio tool slug, e.g. 'GOOGLEDRIVE_DOWNLOAD_FILE'.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments for the tool, matching its input_parameters schema.",
            },
        },
        "required": ["slug", "arguments"],
    },
}


registry.register(
    name="composio_search_tools",
    toolset="composio",
    schema=COMPOSIO_SEARCH_SCHEMA,
    handler=composio_search_tools,
    check_fn=check_composio_available,
    requires_env=["COMPOSIO_API_KEY"],
    emoji="🔗",
    max_result_size_chars=_MAX_RESULT_CHARS,
)
registry.register(
    name="composio_execute_tool",
    toolset="composio",
    schema=COMPOSIO_EXECUTE_SCHEMA,
    handler=composio_execute_tool,
    check_fn=check_composio_available,
    requires_env=["COMPOSIO_API_KEY"],
    emoji="🔗",
    max_result_size_chars=_MAX_RESULT_CHARS,
)
