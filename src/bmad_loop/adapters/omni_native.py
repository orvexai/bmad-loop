"""Omni-owned native coding sessions, driven without an outer multiplexer.

``omni codex`` and ``omni claude`` already create a runner-owned private tmux
terminal.  Wrapping that terminal in another bmad-loop tmux window adds no
ownership and, more importantly, killing the outer window does not prove the
daemon-owned inner session stopped.  This adapter gives each Omni launcher a
small local PTY (only so its attach client has a real terminal), records the
exact Omnigent conversation printed by the launcher, and tears that exact
conversation down before the PTY or its worktree may be retired.

The local receipt at ``tasks/<task-id>/omni-session.json`` is a crash-recovery
contract, not a transcript.  A launch is recorded before process creation; a
hard-killed orchestrator can therefore reconcile the new session by comparing
the stored pre-launch session set with ``omni host status --sessions --json``.
No Omnigent MCP API is used anywhere in this adapter.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import struct
import subprocess
import sys
import termios
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit, urlunsplit

from ..bmadconfig import ProjectPaths
from ..platform_util import atomic_write_text, file_lock
from .generic import GenericAdapter, _DevSynthesisMixin
from .multiplexer import MultiplexerError, TerminalMultiplexer

if TYPE_CHECKING:
    from .base import SessionHandle, SessionSpec


OMNI_COMMAND_TIMEOUT_S = 20.0
OMNI_RECEIPT_TIMEOUT_S = 30.0
OMNI_RECEIPT_POLL_S = 0.2
OMNI_STATUS_POLL_S = 0.5
OMNI_RECEIPT_FILE = "omni-session.json"
OMNI_RECEIPT_SCHEMA = 1

_HARNESS_AGENTS = {
    "claude": "claude-native-ui",
    "codex": "codex-native-ui",
}
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WEB_UI_RE = re.compile(r"Web UI:\s+(https?://[^\s\x1b]+)")


class OmniAdapterError(RuntimeError):
    """The Omni adapter could not establish or safely retire its session."""


@dataclass(frozen=True)
class _HostSession:
    server_url: str
    session_id: str
    workspace: str
    agent_name: str
    runner_online: bool | None

    @property
    def key(self) -> str:
        return f"{self.server_url}\n{_normal_session_id(self.session_id)}"


@dataclass(frozen=True)
class _HostSnapshot:
    sessions: tuple[_HostSession, ...]
    authoritative_servers: frozenset[str]


@dataclass
class _PtyWindow:
    process: subprocess.Popen[bytes]
    master_fd: int
    log_file: Path | None = None
    pending: bytearray = field(default_factory=bytearray)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _OmniPtyHost:
    """The tiny transport Omni needs: one PTY per attached private-tmux client.

    This deliberately is not registered as a general bmad-loop multiplexer.
    It implements only the methods ``GenericAdapter`` calls and exists inside
    one Omni adapter instance.  Omni, not this object, owns the real terminal
    session and its attach/resume semantics.
    """

    def __init__(self) -> None:
        self._sessions: set[str] = set()
        self._geometry: dict[str, tuple[int, int]] = {}
        self._windows: dict[str, _PtyWindow] = {}
        self._lock = threading.Lock()

    def has_session(self, name: str) -> bool:
        return name in self._sessions

    def new_session(
        self, name: str, cwd: Path, cols: int | None = None, lines: int | None = None
    ) -> None:
        del cwd
        self._sessions.add(name)
        self._geometry[name] = (cols or 220, lines or 50)

    def kill_session(self, name: str) -> None:
        for window_id in self.list_window_ids(name):
            self.kill_window(window_id)
        self._sessions.discard(name)

    def list_sessions(self) -> list[str]:
        return sorted(self._sessions)

    def session_options(self, option: str) -> dict[str, str]:
        del option
        return {}

    def set_session_option(self, name: str, option: str, value: str) -> None:
        del name, option, value

    def new_window(
        self, session: str, name: str, cwd: Path, env: dict[str, str], command: str
    ) -> str:
        del name
        argv = shlex.split(command)
        if not argv:
            raise MultiplexerError("Omni PTY launch command was empty")
        window_id = f"omni-pty:{uuid.uuid4().hex}"
        window = self._spawn(session, cwd, env, argv)
        with self._lock:
            self._windows[window_id] = window
        return window_id

    def restart_window(
        self,
        window_id: str,
        session: str,
        cwd: Path,
        env: dict[str, str],
        argv: list[str],
    ) -> None:
        with self._lock:
            prior = self._windows.get(window_id)
        if prior is None:
            raise MultiplexerError(f"unknown Omni PTY window {window_id!r}")
        if prior.process.poll() is None:
            raise MultiplexerError(f"Omni PTY window {window_id!r} is still live")
        replacement = self._spawn(session, cwd, env, argv)
        if prior.log_file is not None:
            self._set_log_file(replacement, prior.log_file)
        with self._lock:
            self._windows[window_id] = replacement

    def _spawn(self, session: str, cwd: Path, env: dict[str, str], argv: list[str]) -> _PtyWindow:
        master_fd, slave_fd = os.openpty()
        cols, lines = self._geometry.get(session, (220, 50))
        try:
            import fcntl

            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", lines, cols, 0, 0))
            process = subprocess.Popen(  # noqa: S603 - argv is shlex-decoded, never a shell
                argv,
                cwd=str(cwd),
                env={**os.environ, **env},
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
            )
        except BaseException:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        os.close(slave_fd)
        window = _PtyWindow(process=process, master_fd=master_fd)
        thread = threading.Thread(
            target=self._drain,
            args=(window,),
            name=f"bmad-loop-omni-pty-{process.pid}",
            daemon=True,
        )
        thread.start()
        return window

    @staticmethod
    def _drain(window: _PtyWindow) -> None:
        try:
            while True:
                try:
                    chunk = os.read(window.master_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                with window.lock:
                    if window.log_file is None:
                        # ``pipe_pane`` follows process creation. Buffering closes
                        # the race where Omni prints its Web UI receipt first.
                        window.pending.extend(chunk)
                    else:
                        try:
                            with window.log_file.open("ab") as fh:
                                fh.write(chunk)
                        except OSError:
                            pass
        finally:
            try:
                os.close(window.master_fd)
            except OSError:
                pass

    def pipe_pane(self, window_id: str, log_file: Path) -> None:
        window = self._window(window_id)
        self._set_log_file(window, log_file)

    @staticmethod
    def _set_log_file(window: _PtyWindow, log_file: Path) -> None:
        with window.lock:
            window.log_file = log_file
            if window.pending:
                try:
                    with log_file.open("ab") as fh:
                        fh.write(window.pending)
                except OSError:
                    pass
                window.pending.clear()

    def list_window_ids(self, session: str) -> list[str]:
        if session not in self._sessions:
            return []
        with self._lock:
            rows = list(self._windows.items())
        return [window_id for window_id, window in rows if window.process.poll() is None]

    def kill_window(self, target: str) -> None:
        try:
            window = self._window(target)
        except MultiplexerError:
            return
        if window.process.poll() is not None:
            return
        try:
            window.process.terminate()
        except (OSError, ProcessLookupError):
            pass

    def send_text(self, window_id: str, text: str) -> None:
        window = self._window(window_id)
        if window.process.poll() is not None:
            raise MultiplexerError(f"Omni PTY window {window_id!r} is not live")
        try:
            os.write(window.master_fd, text.encode("utf-8") + b"\r")
        except OSError as exc:
            raise MultiplexerError(f"could not write to Omni PTY {window_id!r}: {exc}") from exc

    def window_pane_pids(self, target: str) -> list[int]:
        try:
            window = self._window(target)
        except MultiplexerError:
            return []
        return [window.process.pid] if window.process.poll() is None else []

    def _window(self, window_id: str) -> _PtyWindow:
        with self._lock:
            window = self._windows.get(window_id)
        if window is None:
            raise MultiplexerError(f"unknown Omni PTY window {window_id!r}")
        return window


def _normal_session_id(value: str) -> str:
    return value[5:] if value.startswith("conv_") else value


def _same_workspace(left: str, right: str) -> bool:
    try:
        return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
            str(Path(right).resolve())
        )
    except OSError:
        return os.path.normcase(left) == os.path.normcase(right)


def _receipt_from_web_ui(text: str) -> tuple[str, str] | None:
    clean = _ANSI_RE.sub("", text)
    matches = list(_WEB_UI_RE.finditer(clean))
    if not matches:
        return None
    raw_url = matches[-1].group(1).rstrip(".,;)]}")
    parsed = urlsplit(raw_url)
    marker = parsed.path.rfind("/c/")
    if marker < 0:
        return None
    session_id = unquote(parsed.path[marker + 3 :]).strip("/")
    if not session_id:
        return None
    server_path = parsed.path[:marker]
    # Workspace-hosted Omnigent exposes the UI at /omnigent and its API at
    # /api/2.0/omnigent. Direct deployments keep their existing path.
    if server_path == "/omnigent":
        server_path = "/api/2.0/omnigent"
    server_url = urlunsplit((parsed.scheme, parsed.netloc, server_path, "", "")).rstrip("/")
    return server_url, session_id


class OmniNativeAdapter(GenericAdapter):
    """Exact Omni binding, reattach, and stop barrier layered on GenericAdapter."""

    def __init__(self, *args, **kwargs) -> None:
        if sys.platform == "win32":
            raise OmniAdapterError(
                "omni-native requires a POSIX PTY; Omni native harnesses reject Windows"
            )
        profile = kwargs.get("profile")
        if profile is None and len(args) >= 3:
            profile = args[2]
        if profile is None:
            raise OmniAdapterError("omni-native requires a CLI profile")
        launch_args = tuple(profile.launch_args)
        if len(launch_args) != 1 or launch_args[0] not in _HARNESS_AGENTS:
            raise OmniAdapterError(
                "omni-native profile launch_args must contain exactly one harness: "
                "'claude' or 'codex'"
            )
        if profile.hookless:
            raise OmniAdapterError("omni-native requires the underlying Claude/Codex hooks")
        if profile.model_flag != "--model":
            raise OmniAdapterError("omni-native profiles must use model_flag = '--model'")

        self._omni_harness = launch_args[0]
        self._omni_pty = _OmniPtyHost()
        self._omni_specs: dict[str, SessionSpec] = {}
        self._omni_reattach_attempted: set[str] = set()
        kwargs["mux"] = cast(TerminalMultiplexer, self._omni_pty)
        super().__init__(*args, **kwargs)
        self.name = f"{profile.name}-omni-native"
        self._omni_launch_lock = self.run_dir.parents[1] / "omni-native-launch.lock"

    def interactive_argv(self, spec: SessionSpec) -> list[str]:
        extra = self.extra_args if self.extra_args is not None else self.profile.bypass_args
        prompt = self.profile.render_prompt(spec.prompt)
        server = spec.env.get("BMAD_LOOP_OMNI_SERVER") or self.profile.env.get(
            "BMAD_LOOP_OMNI_SERVER"
        )
        prefix = [self.binary, self._omni_harness]
        if server:
            prefix += ["--server", server]
        if self._omni_harness == "codex":
            if spec.model:
                prefix += ["--model", spec.model]
            return [*prefix, "--prompt", prompt, *extra]
        argv = [*prefix, "--prompt", prompt, *extra]
        if spec.model:
            argv += ["--model", spec.model]
        return argv

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        self._omni_specs[spec.task_id] = spec
        task_dir = self.tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self._omni_launch_lock):
            self._reconcile_stale_receipt(spec.task_id)
            baseline = self._host_snapshot()
            receipt: dict[str, Any] = {
                "schema": OMNI_RECEIPT_SCHEMA,
                "task_id": spec.task_id,
                "harness": self._omni_harness,
                "agent_name": _HARNESS_AGENTS[self._omni_harness],
                "workspace": str(spec.cwd.resolve()),
                "state": "launching",
                "launched_ns": time.time_ns(),
                "baseline_sessions": sorted(session.key for session in baseline.sessions),
            }
            self._write_receipt(spec.task_id, receipt)
            handle: SessionHandle | None = None
            try:
                handle = super().start_session(spec)
                bound = self._await_binding(handle, receipt)
                receipt.update(
                    state="bound",
                    server_url=bound.server_url,
                    session_id=bound.session_id,
                    bound_ns=time.time_ns(),
                )
                self._write_receipt(spec.task_id, receipt)
                return handle
            except BaseException as exc:
                receipt["state"] = "launch-failed"
                receipt["error"] = type(exc).__name__
                self._write_receipt(spec.task_id, receipt)
                if handle is not None:
                    try:
                        self._bind_and_stop_after_failed_launch(handle, receipt)
                    finally:
                        GenericAdapter.kill(self, handle)
                raise

    def _await_binding(self, handle: SessionHandle, receipt: dict[str, Any]) -> _HostSession:
        deadline = time.monotonic() + OMNI_RECEIPT_TIMEOUT_S
        next_status = 0.0
        while time.monotonic() < deadline:
            web = self._web_receipt(handle.task_id)
            if web is not None:
                server_url, session_id = web
                return _HostSession(
                    server_url=server_url,
                    session_id=session_id,
                    workspace=str(receipt["workspace"]),
                    agent_name=str(receipt["agent_name"]),
                    runner_online=True,
                )
            now = time.monotonic()
            if now >= next_status:
                candidate = self._single_new_candidate(receipt, self._host_snapshot())
                if candidate is not None:
                    return candidate
                next_status = now + 1.0
            if not GenericAdapter._window_alive(self, handle):
                candidate = self._single_new_candidate(receipt, self._host_snapshot())
                if candidate is not None:
                    return candidate
                raise OmniAdapterError("Omni launcher exited before emitting a session receipt")
            time.sleep(OMNI_RECEIPT_POLL_S)
        raise OmniAdapterError(
            f"Omni did not emit a bindable session receipt within {OMNI_RECEIPT_TIMEOUT_S:g}s"
        )

    def _web_receipt(self, task_id: str) -> tuple[str, str] | None:
        path = self.logs_dir / f"{task_id}.log"
        try:
            with path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 262144))
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        return _receipt_from_web_ui(text)

    def _host_snapshot(self, server: str | None = None) -> _HostSnapshot:
        argv = [self.binary, "host", "status"]
        if server:
            argv += ["--server", server]
        argv += ["--sessions", "--json"]
        result = self._run_omni(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()[:1000]
            raise OmniAdapterError(f"could not inspect Omni host sessions: {detail}")
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise OmniAdapterError("omni host status returned malformed JSON") from exc
        daemons = payload.get("daemons") if isinstance(payload, dict) else None
        if not isinstance(daemons, list):
            raise OmniAdapterError("omni host status JSON omitted daemons[]")
        sessions: list[_HostSession] = []
        authoritative: set[str] = set()
        for daemon in daemons:
            if not isinstance(daemon, dict):
                continue
            server_url = daemon.get("server_url")
            if not isinstance(server_url, str) or not server_url:
                continue
            if daemon.get("process") == "online" and daemon.get("error") in (None, ""):
                authoritative.add(server_url.rstrip("/"))
            rows = daemon.get("sessions")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                session_id = row.get("id")
                workspace = row.get("workspace")
                agent_name = row.get("agent_name")
                if not all(isinstance(v, str) and v for v in (session_id, workspace, agent_name)):
                    continue
                online = row.get("runner_online")
                sessions.append(
                    _HostSession(
                        server_url=server_url.rstrip("/"),
                        session_id=cast(str, session_id),
                        workspace=cast(str, workspace),
                        agent_name=cast(str, agent_name),
                        runner_online=online if isinstance(online, bool) else None,
                    )
                )
        return _HostSnapshot(tuple(sessions), frozenset(authoritative))

    def _single_new_candidate(
        self, receipt: dict[str, Any], snapshot: _HostSnapshot
    ) -> _HostSession | None:
        baseline = {
            value for value in receipt.get("baseline_sessions", []) if isinstance(value, str)
        }
        candidates = [
            session
            for session in snapshot.sessions
            if session.key not in baseline
            and session.agent_name == receipt.get("agent_name")
            and _same_workspace(session.workspace, str(receipt.get("workspace", "")))
        ]
        if len(candidates) > 1:
            receipt["state"] = "ambiguous"
            receipt["candidates"] = [
                {"server_url": row.server_url, "session_id": row.session_id} for row in candidates
            ]
            self._write_receipt(str(receipt["task_id"]), receipt)
            raise OmniAdapterError(
                "multiple new Omni sessions match this task; refusing to guess or stop one"
            )
        return candidates[0] if candidates else None

    def _runner_online(self, server_url: str, session_id: str) -> bool | None:
        snapshot = self._host_snapshot(server_url)
        wanted = _normal_session_id(session_id)
        for session in snapshot.sessions:
            if _normal_session_id(session.session_id) == wanted:
                return session.runner_online
        if server_url.rstrip("/") in snapshot.authoritative_servers:
            return False
        return None

    def _stop_receipt(self, task_id: str, receipt: dict[str, Any]) -> None:
        server_url = receipt.get("server_url")
        session_id = receipt.get("session_id")
        if not isinstance(server_url, str) or not server_url:
            raise OmniAdapterError(f"task {task_id} has no exact Omni server receipt")
        if not isinstance(session_id, str) or not session_id:
            raise OmniAdapterError(f"task {task_id} has no exact Omni session receipt")
        online = self._runner_online(server_url, session_id)
        if online is False:
            receipt.update(state="stopped", stopped_ns=time.time_ns(), stop_evidence="offline")
            self._write_receipt(task_id, receipt)
            return
        receipt["state"] = "stopping"
        self._write_receipt(task_id, receipt)
        result = self._run_omni(
            [self.binary, "host", "stop-session", "--server", server_url, session_id]
        )
        if result.returncode != 0 and self._runner_online(server_url, session_id) is not False:
            detail = (result.stderr or result.stdout or "unknown error").strip()[:1000]
            receipt.update(state="cleanup-failed", error=detail)
            self._write_receipt(task_id, receipt)
            raise OmniAdapterError(f"failed to stop Omni session {session_id}: {detail}")

        deadline = time.monotonic() + max(5.0, float(self.policy.limits.teardown_grace_s))
        while time.monotonic() < deadline:
            if self._runner_online(server_url, session_id) is False:
                receipt.update(
                    state="stopped",
                    stopped_ns=time.time_ns(),
                    stop_evidence="runner-offline",
                )
                receipt.pop("error", None)
                self._write_receipt(task_id, receipt)
                return
            time.sleep(OMNI_STATUS_POLL_S)
        receipt.update(state="cleanup-failed", error="runner remained online after stop-session")
        self._write_receipt(task_id, receipt)
        raise OmniAdapterError(
            f"Omni session {session_id} did not become runner-offline; worktree cleanup is unsafe"
        )

    def _reconcile_stale_receipt(self, task_id: str) -> None:
        receipt = self._read_receipt(task_id)
        if receipt is None or receipt.get("state") in {"stopped", "orphan-not-live"}:
            return
        if isinstance(receipt.get("server_url"), str) and isinstance(
            receipt.get("session_id"), str
        ):
            self._stop_receipt(task_id, receipt)
            return
        snapshot = self._host_snapshot()
        candidate = self._single_new_candidate(receipt, snapshot)
        if candidate is None:
            receipt.update(
                state="orphan-not-live",
                reconciled_ns=time.time_ns(),
                stop_evidence="no-new-matching-live-session",
            )
            self._write_receipt(task_id, receipt)
            return
        receipt.update(server_url=candidate.server_url, session_id=candidate.session_id)
        self._write_receipt(task_id, receipt)
        self._stop_receipt(task_id, receipt)

    def _bind_and_stop_after_failed_launch(
        self, handle: SessionHandle, receipt: dict[str, Any]
    ) -> None:
        web = self._web_receipt(handle.task_id)
        if web is not None:
            receipt.update(server_url=web[0], session_id=web[1], state="bound-after-failure")
            self._write_receipt(handle.task_id, receipt)
            self._stop_receipt(handle.task_id, receipt)
            return
        candidate = self._single_new_candidate(receipt, self._host_snapshot())
        if candidate is None:
            return
        receipt.update(
            server_url=candidate.server_url,
            session_id=candidate.session_id,
            state="bound-after-failure",
        )
        self._write_receipt(handle.task_id, receipt)
        self._stop_receipt(handle.task_id, receipt)

    def _window_alive(self, handle: SessionHandle) -> bool:
        if GenericAdapter._window_alive(self, handle):
            return True
        if handle.task_id in self._omni_reattach_attempted:
            return False
        receipt = self._read_receipt(handle.task_id)
        spec = self._omni_specs.get(handle.task_id)
        if receipt is None or spec is None:
            return False
        server_url = receipt.get("server_url")
        session_id = receipt.get("session_id")
        if not isinstance(server_url, str) or not isinstance(session_id, str):
            return False
        if self._runner_online(server_url, session_id) is not True:
            return False
        self._omni_reattach_attempted.add(handle.task_id)
        try:
            self._omni_pty.restart_window(
                handle.native_id,
                self.session_name,
                spec.cwd,
                self.interactive_env(spec),
                self._resume_argv(spec, server_url, session_id),
            )
        except (OSError, MultiplexerError, subprocess.SubprocessError) as exc:
            receipt.update(state="reattach-failed", error=type(exc).__name__)
            self._write_receipt(handle.task_id, receipt)
            return False
        receipt.update(state="reattached", reattached_ns=time.time_ns())
        self._write_receipt(handle.task_id, receipt)
        return True

    def _resume_argv(self, spec: SessionSpec, server_url: str, session_id: str) -> list[str]:
        extra = self.extra_args if self.extra_args is not None else self.profile.bypass_args
        argv = [
            self.binary,
            self._omni_harness,
            "--server",
            server_url,
            "--resume",
            session_id,
        ]
        if self._omni_harness == "codex" and spec.model:
            argv += ["--model", spec.model]
        argv += list(extra)
        if self._omni_harness == "claude" and spec.model:
            argv += ["--model", spec.model]
        return argv

    def kill(self, handle: SessionHandle) -> None:
        receipt = self._read_receipt(handle.task_id)
        if receipt is None:
            raise OmniAdapterError(
                f"task {handle.task_id} has no Omni receipt; refusing unsafe terminal teardown"
            )
        if receipt.get("state") != "stopped":
            self._stop_receipt(handle.task_id, receipt)
        super().kill(handle)

    def _run_omni(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603 - exact argv, no shell
                argv,
                capture_output=True,
                text=True,
                timeout=OMNI_COMMAND_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OmniAdapterError(
                f"Omni lifecycle command failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _receipt_path(self, task_id: str) -> Path:
        return self.tasks_dir / task_id / OMNI_RECEIPT_FILE

    def _read_receipt(self, task_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._receipt_path(task_id).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_receipt(self, task_id: str, receipt: dict[str, Any]) -> None:
        path = self._receipt_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt["updated_ns"] = time.time_ns()
        atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")


class OmniDevAdapter(_DevSynthesisMixin, OmniNativeAdapter):
    """``bmad-build-auto`` synthesis variant of :class:`OmniNativeAdapter`."""

    def __init__(self, *args, paths: ProjectPaths, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.paths = paths
        self._configure_dev_knobs()
