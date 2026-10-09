import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "webapp"
STATE = WEB / ".deploy"
SECRET_KEYS = {"SPACESHIP_API_KEY", "SPACESHIP_API_SECRET", "CLOUDFLARE_API_TOKEN", "TUNNEL_TOKEN"}


class DeployError(RuntimeError):
    pass


class Pending(DeployError):
    pass


def confirm(message):
    print(message)
    if not sys.stdin.isatty():
        raise Pending("Cần xác nhận tại terminal tương tác; chưa thay đổi tài nguyên.")
    try:
        accepted = input("Nhập yes hoặc YES để xác nhận: ").strip().casefold() == "yes"
    except EOFError:
        accepted = False
    if not accepted:
        raise Pending("Đã hủy thao tác cần xác nhận.")


def private_dir():
    STATE.mkdir(mode=0o700, exist_ok=True)
    if os.name != "nt":
        STATE.chmod(0o700)


def atomic_write(path, data):
    private_dir()
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=STATE)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            Path(path).chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(name, value):
    atomic_write(STATE / name, json.dumps(value, indent=2).encode())


def read_json(name):
    try:
        value = json.loads((STATE / name).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise DeployError("State local hỏng/không đọc được; không tự tạo lại tài nguyên.") from exc
    if not isinstance(value, dict):
        raise DeployError("State local không hợp lệ.")
    return value


class FileLock:
    """Kernel lock; crash releases it, so stale lock files do not block startup."""
    def __init__(self, name):
        self.name = name
        self.stream = None

    def __enter__(self):
        private_dir()
        self.stream = (STATE / self.name).open("a+b")
        self.stream.seek(0)
        if not self.stream.read(1):
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise Pending("Một tiến trình deployment khác đang chạy thao tác này.") from exc
        return self

    def __exit__(self, *args):
        if self.stream:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()


def _dpapi(data, decrypt=False):
    """Windows CurrentUser protection, without an additional Python dependency."""
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    output = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise DeployError("Không đọc/lưu được tunnel token bằng Windows DPAPI của user hiện tại.")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


def save_token(token, tunnel_id):
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise DeployError("Cloudflare trả tunnel token không hợp lệ.")
    # Bind the secret to the tunnel ID to prevent using a stale token with a new state.
    data = json.dumps({"tunnel_id": tunnel_id, "token": token}).encode()
    atomic_write(STATE / "token.bin", b"DPAPI\n" + _dpapi(data) if os.name == "nt" else b"PRIVATE\n" + data)


def load_token(tunnel_id):
    try:
        mode, data = (STATE / "token.bin").read_bytes().split(b"\n", 1)
        if mode == b"DPAPI":
            if os.name != "nt":
                raise ValueError()
            data = _dpapi(data, decrypt=True)
        elif mode != b"PRIVATE" or os.name == "nt":
            raise ValueError()
        secret = json.loads(data)
        if secret["tunnel_id"] != tunnel_id:
            raise ValueError()
        return secret["token"]
    except (OSError, ValueError, KeyError):
        raise Pending("Token cache chưa có/không khớp; chạy setup để lấy token của tunnel cũ.") from None


def read_env(path):
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return {}
    values = {}
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise DeployError(f".env sai cú pháp tại dòng {line_number} (không in giá trị).")
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.rfind(quote)
            if end == 0 or value[end + 1:].strip() and not value[end + 1:].strip().startswith("#"):
                raise DeployError(f".env sai dấu nháy tại dòng {line_number}.")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if "\x00" in value:
            raise DeployError(".env chứa NUL không hợp lệ.")
        values[key] = value
    return values


@dataclass
class Config:
    values: dict
    domain: str
    port: int
    python: str
    account: str
    zone: str

    @classmethod
    def load(cls):
        values = read_env(WEB / ".env")
        values.update({key: value for key, value in os.environ.items() if value})
        domain = values.get("DOMAIN", "spinepredict.top").lower().rstrip(".")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}", domain):
            raise DeployError("DOMAIN phải là hostname DNS hợp lệ.")
        try:
            port = int(values.get("APP_PORT", "5050"))
            if not 1 <= port <= 65535:
                raise ValueError()
        except ValueError:
            raise DeployError("APP_PORT phải là TCP port hợp lệ.") from None
        interpreter = Path(values.get("PYTHON_EXECUTABLE", sys.executable)).expanduser()
        # Preserve the venv executable path: resolving its symlink would launch
        # the base interpreter without this environment's installed packages.
        python = os.path.abspath(interpreter if interpreter.is_absolute() else ROOT / interpreter)
        if not Path(python).is_file():
            raise DeployError("PYTHON_EXECUTABLE không tồn tại; chọn interpreter đã được kiểm chứng.")
        return cls(values, domain, port, python, values.get("CLOUDFLARE_ACCOUNT_ID", "").lower(), values.get("CLOUDFLARE_ZONE_ID", "").lower())

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.port}"

    def require_api(self):
        if not self.values.get("CLOUDFLARE_API_TOKEN"):
            raise Pending("Điền Cloudflare credentials trong webapp/.env; giá trị trong chat không được tự sao chép.")
        if not all(re.fullmatch(r"[0-9a-fA-F]{32}", value) for value in (self.account, self.zone)):
            raise DeployError("Cloudflare account/zone ID phải có 32 ký tự hex.")

    def binding(self):
        return {"domain": self.domain, "port": self.port, "account_id": self.account, "zone_id": self.zone}

    def child_env(self, token=None):
        env = dict(self.values)
        for key in SECRET_KEYS | {"CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_ZONE_ID"}:
            env.pop(key, None)
        env.update(FLASK_HOST="127.0.0.1", FLASK_PORT=str(self.port), PYTHONUNBUFFERED="1")
        if token is not None:
            env["TUNNEL_TOKEN"] = token
        return env
