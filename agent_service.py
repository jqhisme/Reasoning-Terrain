"""Codex execution service used by the FastAPI backend."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import ReasoningSummary, ReasoningSummaryValue
from openai_codex.models import ItemCompletedNotification

from event_formatter import format_completed_item, item_dict
# Tool-call points are temporarily disabled. To restore them, import TOOL_TYPES
# and format_tool_call above, restore logged_tool_calls below, and uncomment the
# ItemStartedNotification block in the event loop.


EventPublisher = Callable[[str, str], None]
SANDBOXES = {
    "read-only": Sandbox.read_only,
    "workspace-write": Sandbox.workspace_write,
    "full-access": Sandbox.full_access,
}


def run_codex_task(
    *,
    task_id: str,
    prompt: str,
    workspace: Path,
    output_dir: Path,
    publish: EventPublisher,
    model: str | None = None,
    provider: str = "openai",
    sandbox: str = "workspace-write",
    network_enabled: bool = True,
    thread_id: str | None = None,
) -> tuple[str, str]:
    """Run one Codex turn synchronously and publish normalized events."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{task_id}.events.jsonl"
    answer_path = output_dir / f"{task_id}.answer.md"

    codex_env = dict(os.environ)
    if not codex_env.get("HOME") and codex_env.get("USERPROFILE"):
        codex_env["HOME"] = codex_env["USERPROFILE"]
    overrides = [
        f"sandbox_workspace_write.network_access={'true' if network_enabled else 'false'}",
        "features.shell_tool=true",
    ]
    if provider == "ollama":
        overrides.append('oss_provider="ollama"')
    config = CodexConfig(
        cwd=str(workspace),
        env=codex_env,
        config_overrides=tuple(overrides),
    )

    final_response: str | None = None
    # logged_tool_calls: set[str] = set()
    with trace_path.open("a", encoding="utf-8", newline="\n") as trace:
        def record(category: str, text: str) -> None:
            if not text:
                return
            trace.write(json.dumps({
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "category": category,
                "text": text,
            }, ensure_ascii=False) + "\n")
            trace.flush()
            publish(category, text)

        with Codex(config) as codex:
            thread_options = {
                "cwd": str(workspace),
                "model": model,
                "model_provider": provider,
                "sandbox": SANDBOXES[sandbox],
                "approval_mode": ApprovalMode.deny_all,
                "developer_instructions": (
                    "Complete the user's task autonomously. Use relevant skills and "
                    "tools, provide concise progress commentary during meaningful "
                    "phases, verify results, and give a clear final answer."
                ),
            }
            if thread_id:
                thread = codex.thread_resume(thread_id, **thread_options)
            else:
                thread = codex.thread_start(**thread_options)
            turn = thread.turn(
                prompt,
                summary=ReasoningSummary(root=ReasoningSummaryValue.detailed),
            )
            for event in turn.stream():
                payload = event.payload
                # Tool-call points are temporarily disabled. Web-search and other
                # completed tools still appear below as function_result points.
                # if isinstance(payload, ItemStartedNotification):
                #     data = item_dict(payload.item)
                #     item_id = data.get("id", "")
                #     if data.get("type") in TOOL_TYPES and item_id not in logged_tool_calls:
                #         text = format_tool_call(data)
                #         if text:
                #             record("tool_call", text)
                #             logged_tool_calls.add(item_id)
                if isinstance(payload, ItemCompletedNotification):
                    data = item_dict(payload.item)
                    for category, text in format_completed_item(payload.item):
                        record(category, text)
                    if data.get("type") == "agentMessage" and data.get("phase") in {
                        "final_answer", None
                    }:
                        final_response = data.get("text")

    if not final_response:
        raise RuntimeError("Codex returned no final answer")
    answer_path.write_text(final_response + "\n", encoding="utf-8")
    return final_response, thread.id
