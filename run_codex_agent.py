"""Run Codex and stream natural-language event types to the visualizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv
from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import ReasoningSummary, ReasoningSummaryValue
from openai_codex.models import ItemCompletedNotification

from event_formatter import format_completed_item, item_dict
# Tool-call points are temporarily disabled. To restore them, import TOOL_TYPES,
# format_tool_call, and ItemStartedNotification, then uncomment the marked block.


HERE = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = HERE.parent
DEFAULT_OUTPUT_DIR = HERE / "runs"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
SANDBOXES = {
    "read-only": Sandbox.read_only,
    "workspace-write": Sandbox.workspace_write,
    "full-access": Sandbox.full_access,
}


def append_record(stream: TextIO, task_id: str, category: str, text: str) -> None:
    stream.write(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "category": category,
                "text": text,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    stream.flush()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def get_task(args: argparse.Namespace) -> str:
    supplied = sum(value is not None for value in (args.task, args.task_file)) + int(
        args.stdin
    )
    if supplied != 1:
        raise SystemExit(
            "Provide exactly one input source: TASK, --task-file, or --stdin."
        )
    if args.task is not None:
        return args.task.strip()
    if args.task_file is not None:
        return args.task_file.read_text(encoding="utf-8").strip()
    return sys.stdin.read().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Codex task and visualize its semantic event landscape."
    )
    parser.add_argument("task", metavar="TASK", nargs="?")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", help="Optional model override")
    parser.add_argument(
        "--provider", choices=("openai", "ollama"), default="openai"
    )
    parser.add_argument(
        "--sandbox", choices=SANDBOXES, default="workspace-write"
    )
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="Write normalized events locally without posting them",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    task = get_task(args)
    if not task:
        raise SystemExit("Task text cannot be empty.")

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist: {workspace}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    trace_path = output_dir / f"{run_id}.events.jsonl"
    answer_path = output_dir / f"{run_id}.answer.md"
    model = args.model or (
        DEFAULT_OLLAMA_MODEL if args.provider == "ollama" else None
    )

    if args.no_backend:
        task_id = run_id
    else:
        try:
            state = post_json(
                f"{args.backend_url.rstrip('/')}/api/tasks",
                {"task_id": run_id, "prompt": task},
            )
            task_id = state["task_id"]
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(
                f"Backend unavailable at {args.backend_url}. Start it first or pass "
                f"--no-backend. Details: {exc}"
            ) from exc

    network_enabled = not args.no_network
    codex_env = dict(os.environ)
    if not codex_env.get("HOME") and codex_env.get("USERPROFILE"):
        codex_env["HOME"] = codex_env["USERPROFILE"]
    config_overrides = [
        f"sandbox_workspace_write.network_access={'true' if network_enabled else 'false'}",
        "features.shell_tool=true",
    ]
    if args.provider == "ollama":
        config_overrides.append('oss_provider="ollama"')
    config = CodexConfig(
        cwd=str(workspace),
        env=codex_env,
        config_overrides=tuple(config_overrides),
    )

    final_response: str | None = None
    # logged_tool_calls: set[str] = set()

    with trace_path.open("w", encoding="utf-8", newline="\n") as trace:

        def publish(category: str, text: str) -> None:
            if not text:
                return
            append_record(trace, task_id, category, text)
            if not args.no_backend:
                post_json(
                    f"{args.backend_url.rstrip('/')}/api/events",
                    {"task_id": task_id, "category": category, "text": text},
                )

        try:
            with Codex(config) as codex:
                thread = codex.thread_start(
                    cwd=str(workspace),
                    model=model,
                    model_provider=args.provider,
                    sandbox=SANDBOXES[args.sandbox],
                    approval_mode=ApprovalMode.deny_all,
                    developer_instructions=(
                        "Complete the user's task autonomously. Use relevant skills and "
                        "tools, provide concise progress commentary during meaningful "
                        "phases, verify results, and give a clear final answer."
                    ),
                )
                turn = thread.turn(
                    task,
                    summary=ReasoningSummary(root=ReasoningSummaryValue.detailed),
                )
                for event in turn.stream():
                    payload = event.payload
                    # Tool-call points are temporarily disabled. Completed web
                    # searches still publish as function_result points below.
                    # if isinstance(payload, ItemStartedNotification):
                    #     data = item_dict(payload.item)
                    #     if data.get("type") in TOOL_TYPES:
                    #         text = format_tool_call(data)
                    #         item_id = data.get("id", "")
                    #         if text and item_id not in logged_tool_calls:
                    #             publish("tool_call", text)
                    #             logged_tool_calls.add(item_id)
                    if isinstance(payload, ItemCompletedNotification):
                        data = item_dict(payload.item)
                        for category, text in format_completed_item(payload.item):
                            publish(category, text)
                        if data.get("type") == "agentMessage":
                            phase = data.get("phase")
                            if phase == "final_answer" or phase is None:
                                final_response = data.get("text")
        except Exception:
            print(f"Codex run failed. Partial log: {trace_path}", file=sys.stderr)
            raise

    if not final_response:
        raise SystemExit(f"Codex returned no final answer. Inspect {trace_path}")
    answer_path.write_text(final_response + "\n", encoding="utf-8")
    if not args.no_backend:
        post_json(
            f"{args.backend_url.rstrip('/')}/api/final-answer",
            {"task_id": task_id, "text": final_response},
        )

    print(final_response)
    print(f"\nNatural-language event log: {trace_path}")
    print(f"Final answer: {answer_path}")
    if not args.no_backend:
        print(f"Visualization: {args.backend_url.rstrip('/')}/?task={task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
