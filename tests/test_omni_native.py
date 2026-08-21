"""Omni-native adapter lifecycle and no-outer-tmux proofs."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bmad_loop.adapters.base import SessionHandle, SessionSpec
from bmad_loop.adapters.omni_native import (
    OmniAdapterError,
    OmniNativeAdapter,
    _OmniPtyHost,
    _receipt_from_web_ui,
)
from bmad_loop.adapters.profile import get_profile
from bmad_loop.policy import LimitsPolicy, Policy


def _adapter(tmp_path: Path, profile: str = "omni-codex") -> OmniNativeAdapter:
    run_dir = tmp_path / ".bmad-loop" / "runs" / "r1"
    return OmniNativeAdapter(
        run_dir=run_dir,
        policy=Policy(limits=LimitsPolicy(teardown_grace_s=0)),
        profile=get_profile(profile),
        events_dir=tmp_path / "events",
    )


def _spec(tmp_path: Path, model: str = "gpt-5.6-luna") -> SessionSpec:
    return SessionSpec(
        task_id="story-1-dev-1",
        role="dev",
        prompt="/bmad-build-auto story-1",
        cwd=tmp_path,
        env={"BMAD_LOOP_TASK_ID": "story-1-dev-1"},
        model=model,
    )


def test_omni_profiles_build_harness_specific_argv(tmp_path):
    codex = _adapter(tmp_path, "omni-codex")
    codex_argv = codex.interactive_argv(_spec(tmp_path))
    assert codex_argv[:2] == ["omni", "codex"]
    assert codex_argv.index("--model") < codex_argv.index("--prompt")
    assert "--dangerously-bypass-approvals-and-sandbox" in codex_argv

    claude = _adapter(tmp_path, "omni-claude")
    claude_argv = claude.interactive_argv(_spec(tmp_path, model="opus"))
    assert claude_argv[:2] == ["omni", "claude"]
    assert claude_argv.index("--prompt") < claude_argv.index("--model")
    assert "--use-native-config" not in claude_argv
    assert claude_argv[-2:] == ["--model", "opus"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "\x1b[32mWeb UI: https://agents.example/c/abc123\x1b[0m\r\n",
            ("https://agents.example", "abc123"),
        ),
        (
            "Web UI: https://workspace.example/omnigent/c/conv_abc123?o=42\n",
            ("https://workspace.example/api/2.0/omnigent", "conv_abc123"),
        ),
    ],
)
def test_web_ui_line_is_an_exact_stop_receipt(line, expected):
    assert _receipt_from_web_ui(line) == expected


def test_host_snapshot_is_machine_readable_and_server_scoped(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    seen: list[list[str]] = []
    payload = {
        "daemons": [
            {
                "server_url": "https://agents.example",
                "process": "online",
                "error": None,
                "sessions": [
                    {
                        "id": "abc",
                        "workspace": str(tmp_path),
                        "agent_name": "codex-native-ui",
                        "runner_online": True,
                    }
                ],
            }
        ]
    }

    def fake_run(argv):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(adapter, "_run_omni", fake_run)
    snapshot = adapter._host_snapshot("https://agents.example")
    assert seen == [
        [
            "omni",
            "host",
            "status",
            "--server",
            "https://agents.example",
            "--sessions",
            "--json",
        ]
    ]
    assert snapshot.authoritative_servers == {"https://agents.example"}
    assert snapshot.sessions[0].session_id == "abc"
    assert snapshot.sessions[0].runner_online is True


def test_exact_stop_waits_for_runner_offline_before_marking_receipt(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    states = iter((True, True, False))
    commands: list[list[str]] = []
    monkeypatch.setattr(adapter, "_runner_online", lambda *_: next(states))

    def fake_run(argv):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "Stopped session abc.\n", "")

    monkeypatch.setattr(adapter, "_run_omni", fake_run)
    monkeypatch.setattr("bmad_loop.adapters.omni_native.OMNI_STATUS_POLL_S", 0)
    receipt = {
        "task_id": "story-1-dev-1",
        "state": "bound",
        "server_url": "https://agents.example",
        "session_id": "abc",
    }
    adapter._stop_receipt("story-1-dev-1", receipt)
    assert commands == [
        [
            "omni",
            "host",
            "stop-session",
            "--server",
            "https://agents.example",
            "abc",
        ]
    ]
    persisted = adapter._read_receipt("story-1-dev-1")
    assert persisted["state"] == "stopped"
    assert persisted["stop_evidence"] == "runner-offline"


def test_cleanup_blocks_when_stop_does_not_take_runner_offline(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr(adapter, "_runner_online", lambda *_: True)
    monkeypatch.setattr(
        adapter,
        "_run_omni",
        lambda argv: subprocess.CompletedProcess(argv, 0, "Stopped session abc.\n", ""),
    )
    ticks = iter((0.0, 0.0, 6.0))
    monkeypatch.setattr("bmad_loop.adapters.omni_native.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("bmad_loop.adapters.omni_native.time.sleep", lambda _: None)
    receipt = {
        "task_id": "story-1-dev-1",
        "state": "bound",
        "server_url": "https://agents.example",
        "session_id": "abc",
    }
    with pytest.raises(OmniAdapterError, match="worktree cleanup is unsafe"):
        adapter._stop_receipt("story-1-dev-1", receipt)
    assert adapter._read_receipt("story-1-dev-1")["state"] == "cleanup-failed"


def test_pty_host_supplies_tty_and_captures_output_without_outer_tmux(tmp_path):
    host = _OmniPtyHost()
    host.new_session("run", tmp_path, 100, 30)
    log = tmp_path / "omni.log"
    log.touch()
    script = (
        "import os, time; assert os.isatty(0); "
        "print('Web UI: https://agents.example/c/abc', flush=True); time.sleep(30)"
    )
    command = shlex.join([sys.executable, "-c", script])
    window_id = host.new_window("run", "story", tmp_path, {}, command)
    try:
        host.pipe_pane(window_id, log)
        deadline = time.monotonic() + 5
        while "Web UI:" not in log.read_text(encoding="utf-8", errors="replace"):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert window_id in host.list_window_ids("run")
    finally:
        host.kill_window(window_id)
        deadline = time.monotonic() + 5
        while window_id in host.list_window_ids("run") and time.monotonic() < deadline:
            time.sleep(0.02)


def test_dead_attach_client_uses_native_harness_resume_once(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    spec = _spec(tmp_path)
    adapter._omni_specs[spec.task_id] = spec
    adapter._write_receipt(
        spec.task_id,
        {
            "state": "bound",
            "server_url": "https://agents.example",
            "session_id": "abc",
        },
    )
    monkeypatch.setattr(adapter._omni_pty, "list_window_ids", lambda _: [])
    monkeypatch.setattr(adapter, "_runner_online", lambda *_: True)
    restarts: list[list[str]] = []

    def restart(window_id, session, cwd, env, argv):
        del window_id, session, cwd, env
        restarts.append(argv)

    monkeypatch.setattr(adapter._omni_pty, "restart_window", restart)
    handle = SessionHandle(task_id=spec.task_id, native_id="omni-pty:1")
    assert adapter._window_alive(handle) is True
    assert restarts == [
        [
            "omni",
            "codex",
            "--server",
            "https://agents.example",
            "--resume",
            "abc",
            "--model",
            "gpt-5.6-luna",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
    ]
    # One recovery attempt per task: a repeatedly dying attach becomes a
    # terminal failure and is stopped by kill(), not an infinite resume loop.
    assert adapter._window_alive(handle) is False
