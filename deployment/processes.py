"""A single detached supervisor owns native processes; normal start makes no API writes."""
import ctypes
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request

from .common import ROOT, ENTRYPOINT, STATE, SECRET_KEYS, Config, DeployError, FileLock, Pending, load_token, private_dir, read_json, write_json
from .setup import require_state
from .transport import network_reason, urlopen


def stamp(pid):
    if not isinstance(pid, int) or pid < 1:
        return None
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                return None
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    if sys.platform == "darwin":
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
        return result.stdout.strip() or None
    try:  # Also usable in a Linux test environment; deployment targets remain Win/macOS.
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if parts[0] == "Z" else parts[19]
    except OSError:
        return None


def alive(record):
    return bool(record and record.get("stamp") and stamp(record.get("pid")) == record["stamp"])


def terminate(record):
    """Only kill a saved owned process whose creation timestamp still matches."""
    if not record.get("owned") or not alive(record):
        return
    pid = record["pid"]
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
            time.sleep(1)
            # The group was verified above; kill remaining worker descendants too.
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can refuse signalling a group after its leader has exited.
            # Never target a reused/dead PID; only fall back for the same live root.
            if alive(record):
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if alive(record):
                    os.kill(pid, signal.SIGKILL)


def get_json(url, timeout=5, values=None, headers=None, diagnostics=None):
    details = diagnostics if diagnostics is not None else {}
    try:
        request = Request(url, headers=headers) if headers else url
        with urlopen(request, timeout=timeout, values=values) as response:
            try:
                body = json.load(response)
                if not isinstance(body, dict):
                    raise ValueError()
            except ValueError:
                details["reason"] = "invalid-response"
                body = {}
            return response.status, body, response.geturl()
    except HTTPError as exc:
        details["reason"] = "http-error"
        details["cf_ray"] = exc.headers.get("CF-Ray")
        return exc.code, {}, exc.geturl()
    except OSError as exc:
        details["reason"] = network_reason(exc)
        return None, {}, url
    except ValueError:
        details["reason"] = "invalid-response"
        return None, {}, url


def readiness(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/ready", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def app_health(config):
    status, body, _ = get_json(config.origin + "/api/health", timeout=2, values=config.values)
    return status == 200 and body.get("model_loaded") is True and body.get("model", {}).get("bundle_type") == "fusion_subsidence_full_late_fusion"


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def preflight(config):
    code = """import json, sys
try:
    if sys.version_info < (3, 10):
        raise RuntimeError('Python >= 3.10 required')
    from webapp.app import app
    if app.extensions['late_fusion_predictor'] is None:
        raise RuntimeError(app.extensions.get('model_load_error') or 'Model bundle unavailable')
except Exception as exc:
    print('DEPLOY_PREFLIGHT_ERROR=' + json.dumps({'type': type(exc).__name__, 'module': getattr(exc, 'name', None), 'message': str(exc)}))
    sys.exit(1)
"""
    try:
        result = subprocess.run([config.python, "-c", code], cwd=ROOT, env=config.child_env(), capture_output=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired):
        raise DeployError("Không kiểm tra được interpreter; kiểm tra Python/dependencies/model tại máy này.") from None
    if result.returncode:
        detail = "Không import/load được app; kiểm tra Python và model bundle."
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            if line.startswith("DEPLOY_PREFLIGHT_ERROR="):
                try:
                    failure = json.loads(line.split("=", 1)[1])
                    if failure.get("type") == "ModuleNotFoundError":
                        detail = "Thiếu thư viện Python: " + str(failure.get("module"))
                    else:
                        detail = str(failure.get("type")) + ": " + str(failure.get("message"))
                except (ValueError, AttributeError):
                    pass
        for key in SECRET_KEYS:
            secret = config.values.get(key)
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        detail = detail.replace("\n", " ").replace("\r", " ")[:600]
        raise DeployError(f"Python preflight thất bại ({config.python}): {detail}")
    print("PASS: Python và model import/load; MRI workers/SHAP thực tế vẫn cần nghiệm thu riêng.")


def cloudflared(config):
    """Use a shared installation; never download a project-local binary."""
    configured = config.values.get("CLOUDFLARED_PATH")
    binary = str(Path(configured).expanduser().resolve()) if configured else shutil.which("cloudflared")
    # Existing terminals may still have the PATH from before installation.
    if not binary and os.name == "nt":
        for variable, subdirectory in (
            ("LOCALAPPDATA", "Programs/cloudflared"),
            ("ProgramFiles", "cloudflared"),
            ("ProgramFiles(x86)", "cloudflared"),
        ):
            base = os.environ.get(variable)
            if not base:
                continue
            for suffix in ("cloudflared.exe", "bin/cloudflared.exe"):
                candidate = Path(base) / subdirectory / suffix
                if candidate.is_file():
                    binary = str(candidate)
                    break
            if binary:
                break
    if not binary:
        raise Pending(
            "Ch?a t?m th?y cloudflared global. C?i b?ng winget install --id Cloudflare.cloudflared "
            "--exact (Windows) ho?c brew install cloudflared (macOS), r?i m? terminal m?i. "
            "C? th? ??t CLOUDFLARED_PATH t?i b?n ?? c?i; kh?ng t? t?i v?o d? ?n."
        )
    try:
        result = subprocess.run([binary, "version"], capture_output=True, timeout=10)
        if result.returncode == 0 and b"cloudflared version" in result.stdout + result.stderr:
            return binary
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise DeployError("CLOUDFLARED_PATH/binary kh?ng h?p l?; ki?m tra b?n cloudflared ?? c?i.")


def logger(config, token):
    private_dir()
    log = logging.getLogger("spinepredict.deploy")
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(STATE / "process.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    class Redact(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            for secret in [token, *[config.values.get(key, "") for key in SECRET_KEYS]]:
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            message = re.sub(r"/api/extraction-files/[^\s?\"']+", "/api/extraction-files/[REDACTED]", message)
            record.msg, record.args = message, ()
            return True

    handler.addFilter(Redact())
    log.addHandler(handler)
    if os.name != "nt":
        (STATE / "process.log").chmod(0o600)
    return log


def spawn(command, env, log, kind):
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, **options)

    def capture():
        with process.stdout:
            while True:
                line = process.stdout.readline(65536)
                if not line:
                    break
                log.info("%s: %s", kind, line.decode("utf-8", errors="replace").strip())

    threading.Thread(target=capture, daemon=True).start()
    return process, {"pid": process.pid, "stamp": stamp(process.pid), "owned": True}


def supervise(config, binary):
    # An explicit stop also prevents an OS startup task from reviving the app.
    if (STATE / "stopped.json").is_file():
        return
    state = require_state(config)
    token = load_token(state["tunnel_id"])
    with FileLock("runtime.lock"):
        if (STATE / "stopped.json").is_file():
            return
        old = read_json("runtime.json")
        # Clean only saved owned children from a crashed supervisor, never an external app.
        for key in ("tunnel", "app"):
            terminate(old.get(key, {}))
        stop_file = STATE / "stop.json"
        stop_file.unlink(missing_ok=True)
        log = logger(config, token)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            metrics_port = listener.getsockname()[1]
        runtime = {"pid": os.getpid(), "stamp": stamp(os.getpid()), "nonce": uuid.uuid4().hex,
                   "owned": True, "metrics_port": metrics_port, "tunnel_id": state["tunnel_id"]}
        children, due, failures = {}, {"app": 0, "tunnel": 0}, {"app": 0, "tunnel": 0}
        running = True

        def shutdown(*args):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        try:
            if port_open(config.port):
                if not app_health(config):
                    raise DeployError("Port đã bị chiếm bởi service khác/app degraded. Không kill hoặc tạo app trùng.")
                runtime["app_external"] = True
                log.info("Reuse healthy external Python app; stop/restart will not kill it.")
            while running:
                if (STATE / "stopped.json").is_file():
                    break
                if stop_file.is_file() and read_json("stop.json").get("nonce") == runtime["nonce"]:
                    break
                now = time.monotonic()
                for kind, process in list(children.items()):
                    if process.poll() is not None:
                        log.info("%s exited; restart with backoff.", kind)
                        # Descendants can remain after root process exits. They are not
                        # force-killed from an unverified dead PID/group.
                        children.pop(kind)
                        runtime.pop(kind, None)
                        failures[kind] += 1
                        due[kind] = now + min(60, 2 ** min(failures[kind], 6))
                healthy = app_health(config)
                if "app" not in children and not runtime.get("app_external") and now >= due["app"]:
                    if port_open(config.port):
                        raise DeployError("Port bị chiếm trong lúc restart; không tạo app trùng.")
                    children["app"], runtime["app"] = spawn([config.python, "-m", "webapp.app"], config.child_env(), log, "app")
                if runtime.get("app_external") and not port_open(config.port):
                    runtime.pop("app_external")
                if healthy and "tunnel" not in children and now >= due["tunnel"]:
                    command = [binary, "tunnel", "--no-autoupdate", "--loglevel", "info", "--metrics",
                               f"127.0.0.1:{metrics_port}", "run"]
                    children["tunnel"], runtime["tunnel"] = spawn(command, config.child_env(token), log, "tunnel")
                runtime["heartbeat"] = time.time()
                runtime["app_healthy"] = healthy
                write_json("runtime.json", runtime)
                time.sleep(2)
        except Exception as exc:
            detail = str(exc) if isinstance(exc, DeployError) else type(exc).__name__
            log.error("Supervisor failed: %s", detail)
            raise
        finally:
            for key in ("tunnel", "app"):
                terminate(runtime.get(key, {}))
            for process in children.values():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            runtime["stopped"] = True
            write_json("runtime.json", runtime)
            log.info("Supervisor stopped; retained model/data/output files unchanged.")


def start(config, preflight_done=False):
    with FileLock("command.lock"):
        require_state(config)  # No setup/API mutations inside start or autostart.
        runtime = read_json("runtime.json")
        if alive(runtime) and not runtime.get("stopped"):
            print("Supervisor đã chạy; không tạo tiến trình trùng.")
            return
        binary = cloudflared(config)
        if not preflight_done:
            preflight(config)
        if port_open(config.port) and not app_health(config):
            raise DeployError("Port đang bị chiếm/app unhealthy. Không tự kill service hiện có.")
        options = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        from .autostart import launch_registered
        (STATE / "stopped.json").unlink(missing_ok=True)
        registered = launch_registered()
        supervisor = None if registered else subprocess.Popen(
            [config.python, str(ENTRYPOINT), "_supervise", "--binary", binary],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, **options)
        for _ in range(50):
            current = read_json("runtime.json")
            if (alive(current) and not current.get("stopped")
                    and alive(current.get("tunnel")) and app_health(config)):
                print("Python healthy; Named Tunnel connector đã khởi động.")
                print("Public URL dự kiến: https://" + config.domain + " (dùng verify để xác nhận HTTPS/propagation).")
                return
            if supervisor is not None and supervisor.poll() is not None:
                raise DeployError(f"Supervisor thoát; kiểm tra {STATE / 'process.log'} và port/config.")
            time.sleep(1)
        raise Pending("Supervisor đang khởi động; dùng status/verify, không start app/tunnel khác.")


def stop(config):
    with FileLock("command.lock"):
        write_json("stopped.json", {"stopped": True})
        runtime = read_json("runtime.json")
        if alive(runtime) and not runtime.get("stopped"):
            write_json("stop.json", {"nonce": runtime["nonce"]})
            for _ in range(20):
                if not alive(runtime):
                    break
                time.sleep(1)
            else:
                current = read_json("runtime.json")
                terminate(runtime)
                for key in ("tunnel", "app"):
                    terminate(current.get(key, {}))
        else:
            for key in ("tunnel", "app"):
                terminate(runtime.get(key, {}))
        print("Đã dừng deployment; không tự bật lại cho tới khi chạy run/start. App ngoài deployment không bị kill.")


def status(config):
    runtime = read_json("runtime.json")
    healthy = app_health(config)
    print("Supervisor:", "RUNNING" if alive(runtime) and not runtime.get("stopped") else "STOPPED")
    print("Local app:", "HEALTHY" if healthy else "UNAVAILABLE/DEGRADED")
    ownership = "managed" if alive(runtime.get("app")) else "external/unmanaged (không bị stop/restart)" if healthy else "none"
    print("App ownership:", ownership)
    print("Managed tunnel process:", "RUNNING" if alive(runtime.get("tunnel")) else "STOPPED")
    print("URL:", "https://" + config.domain)
