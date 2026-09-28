"""Convert completed Codex items into human-readable event categories."""

from __future__ import annotations

import json
from typing import Any


CATEGORIES = {
    "reasoning_summary",
    "plan",
    "progress_update",
    # Temporarily disabled. Uncomment when tool-call points return to the UI.
    # "tool_call",
    "function_result",
}

TOOL_TYPES = {
    "commandExecution",
    "webSearch",
    "mcpToolCall",
    "dynamicToolCall",
    "collabToolCall",
    "imageView",
    "fileChange",
}


def _short(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def item_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "root"):
        item = item.root
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", by_alias=True)
    if isinstance(item, dict):
        return item
    return {}


def format_tool_call(item: dict[str, Any]) -> str | None:
    kind = item.get("type")
    if kind == "commandExecution":
        command = _short(item.get("command"))
        cwd = item.get("cwd")
        location = f" in {cwd}" if cwd else ""
        return f"Using the shell tool to execute `{command}`{location}."
    if kind == "webSearch":
        query = item.get("query") or (item.get("action") or {}).get("query")
        return f"Using web search to research: {_short(query)}."
    if kind == "mcpToolCall":
        name = ".".join(filter(None, [item.get("server"), item.get("tool")]))
        return f"Using the MCP tool {name} with arguments {_short(item.get('arguments'))}."
    if kind == "dynamicToolCall":
        return (
            f"Using the tool {item.get('tool', 'unknown')} with arguments "
            f"{_short(item.get('arguments'))}."
        )
    if kind == "collabToolCall":
        return f"Using the collaboration tool {item.get('tool', 'unknown')}."
    if kind == "imageView":
        return f"Using the image viewer to inspect {_short(item.get('path'))}."
    if kind == "fileChange":
        paths = [change.get("path") for change in item.get("changes", [])]
        return f"Using the file-editing tool to change: {_short(paths)}."
    return None


def format_tool_result(item: dict[str, Any]) -> str | None:
    kind = item.get("type")
    if kind == "commandExecution":
        output = _short(item.get("aggregatedOutput")) or "No output was returned"
        return (
            f"The shell command finished with status {item.get('status', 'unknown')} "
            f"and exit code {item.get('exitCode')}. Result: {output}"
        )
    if kind == "webSearch":
        query = item.get("query") or (item.get("action") or {}).get("query")
        return f"Web search completed for: {_short(query)}."
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        tool = item.get("tool", "unknown")
        result = item.get("result") or item.get("contentItems") or item.get("error")
        success = item.get("success", item.get("status", "unknown"))
        return f"Tool {tool} completed with status {success}. Result: {_short(result)}"
    if kind == "collabToolCall":
        return (
            f"Collaboration tool {item.get('tool', 'unknown')} completed with "
            f"status {item.get('status', 'unknown')}."
        )
    if kind == "imageView":
        return f"The image viewer finished inspecting {_short(item.get('path'))}."
    if kind == "fileChange":
        return f"File changes completed with status {item.get('status', 'unknown')}."
    return None


def format_completed_item(item: Any) -> list[tuple[str, str]]:
    data = item_dict(item)
    kind = data.get("type")
    blocks: list[tuple[str, str]] = []

    if kind == "reasoning":
        for summary in data.get("summary") or []:
            if summary:
                blocks.append(("reasoning_summary", _short(summary)))
    elif kind == "plan" and data.get("text"):
        blocks.append(("plan", _short(data["text"])))
    elif kind == "agentMessage" and data.get("phase") == "commentary":
        blocks.append(("progress_update", _short(data.get("text"))))
    elif kind in TOOL_TYPES:
        result = format_tool_result(data)
        if result:
            blocks.append(("function_result", result))
    elif kind in {"functionCallOutput", "mcpToolCallOutput", "dynamicToolCallOutput"}:
        blocks.append(
            (
                "function_result",
                f"Function {data.get('name', kind)} returned: {_short(data.get('output'))}",
            )
        )
    return blocks
