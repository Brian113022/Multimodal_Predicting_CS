"""Run an extractor helper script in the environment that owns its dependencies.

Adapters hand a JSON job to a helper script and read back stage events and a
result, one JSON object per line. Helpers can use the web app's interpreter or
a configured interpreter with compatible imaging dependencies.
"""

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .errors import ExtractionExecutionError, ExtractorUnavailableError


def terminate_worker(process):
    """Stop a registered feature worker and its child CLI processes."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        else:
            # Registered workers start in their own session, so this group
            # contains only that worker and its extraction children.
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()


def resolve_python(configured: str, sibling_command: str = "") -> str:
    """Pick the interpreter for a helper environment.

    ``configured`` wins. Otherwise the environment is derived from a command
    that lives in it (``<env>/Scripts/x.exe`` or ``<env>/bin/x``), and finally
    the current interpreter is used.
    """
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise ExtractorUnavailableError(
                f"Configured interpreter does not exist: {configured_path}"
            )
        return str(configured_path.resolve())

    if sibling_command:
        command_directory = Path(sibling_command).parent
        for candidate in (
            command_directory.parent / "python.exe",
            command_directory / "python",
            command_directory.parent / "bin" / "python",
        ):
            if candidate.is_file():
                return str(candidate.resolve())
    return sys.executable


def run_worker(
        python_executable: str,
        script: Path,
        job: dict,
        *,
        stage_prefix: str,
        result_prefix: str,
        timeout_seconds: int,
        label: str,
        on_process=None,
):
    """Run ``script``, yielding its stage events and returning its report.

    ``on_process`` receives the started process, so a caller that runs several
    workers at once can kill them.
    """
    if not script.is_file():
        raise ExtractorUnavailableError(f"{label} worker script is missing: {script}")
    if not python_executable or not Path(python_executable).is_file():
        resolved = shutil.which(python_executable or "python")
        if resolved is None:
            raise ExtractorUnavailableError(
                f"No interpreter is available to run the {label} worker."
            )
        python_executable = resolved

    try:
        process = subprocess.Popen(
            [python_executable, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=on_process is not None and os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if on_process is not None and os.name == "nt" else 0,
        )
    except OSError as exc:
        raise ExtractorUnavailableError(
            f"Cannot start the {label} worker with {python_executable}: {exc}"
        ) from exc

    if on_process is not None:
        try:
            on_process(process)
        except BaseException:
            terminate_worker(process)
            raise

    # stderr is drained on its own thread. Left unread until stdout closed, a
    # worker that logs more than the pipe buffer (4 KB on Windows; the full
    # 1,688-feature PyRadiomics run already writes 3.5 KB of warnings) blocks
    # on the write and never closes stdout.
    stderr_chunks = []

    def drain_stderr():
        try:
            stderr_chunks.append(process.stderr.read())
        except (OSError, ValueError):
            # The stream was closed below because the caller stopped early.
            pass

    stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
    stderr_reader.start()

    payload = None
    try:
        process.stdin.write(json.dumps(job))
        process.stdin.close()
        # Read as the worker prints so stages surface while it is still running.
        for line in process.stdout:
            line = line.strip()
            if line.startswith(stage_prefix):
                try:
                    yield json.loads(line[len(stage_prefix):])
                except ValueError:
                    pass
            elif line.startswith(result_prefix):
                payload = line[len(result_prefix):]
        # The worker bounds its own heavy calls; this only bounds its shutdown.
        process.wait(timeout=120)
        stderr_reader.join(timeout=10)
        stderr_text = "".join(stderr_chunks)
    except subprocess.TimeoutExpired as exc:
        if on_process is not None:
            terminate_worker(process)
        else:
            process.kill()
        raise ExtractionExecutionError(
            f"{label} extraction exceeded the {timeout_seconds}-second timeout."
        ) from exc
    finally:
        if process.poll() is None:
            if on_process is not None:
                terminate_worker(process)
            else:
                process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except Exception:
                pass

    if payload is None:
        diagnostic = diagnostic_tail(stderr_text)
        if process.returncode != 0:
            raise ExtractorUnavailableError(
                f"The {label} worker exited with code {process.returncode}. "
                f"Check that its interpreter has the required packages: {diagnostic}"
            )
        raise ExtractionExecutionError(
            f"The {label} worker returned no result: {diagnostic}"
        )

    try:
        report = json.loads(payload)
    except ValueError as exc:
        raise ExtractionExecutionError(
            f"The {label} worker returned malformed output: {exc}"
        ) from exc

    if report.get("status") != "ok":
        raise ExtractionExecutionError(
            report.get("message") or f"{label} extraction failed."
        )
    return report


def diagnostic_tail(value: str, limit: int = 1500) -> str:
    compact = " ".join((value or "").split())
    return compact[-limit:] or "no diagnostic output"
