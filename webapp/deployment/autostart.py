"""Opt-in user startup. System configuration changes always require confirmation."""
import hashlib
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from xml.etree import ElementTree as ET

from .common import ROOT, WEB, STATE, DeployError, Pending, confirm, read_json, write_json, atomic_write
from .processes import alive, cloudflared
from .setup import require_state


def checked(command):
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise DeployError("OS không chấp nhận thao tác auto-start; kiểm tra quyền/policy và hướng dẫn DEPLOYMENT.md.")


def install(config):
    require_state(config)
    binary = cloudflared(config)
    cached = read_json("autostart.json")
    if cached:
        raise Pending("Auto-start đã có cấu hình local; kiểm tra status hoặc uninstall trước khi cài lại.")
    runtime = read_json("runtime.json")
    if alive(runtime) and not runtime.get("stopped"):
        raise Pending("Chạy stop trước khi cài auto-start để tránh hai supervisor tranh lock.")
    identity = hashlib.sha256((str(ROOT) + config.python).encode()).hexdigest()[:12]
    if os.name == "nt":
        name = "SpinePredict-" + identity
        if subprocess.run(["schtasks", "/Query", "/TN", name], capture_output=True).returncode == 0:
            raise Pending("Tên task đã tồn tại; không ghi đè task không được quản lý.")
        confirm("Cài Task Scheduler cho user hiện tại: app/tunnel chạy sau đăng nhập, retry crash mỗi phút? Không lưu mật khẩu.")
        user = subprocess.check_output(["whoami"], text=True).strip()
        namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
        ET.register_namespace("", namespace)

        def child(parent, name, value=None, **attrs):
            node = ET.SubElement(parent, "{" + namespace + "}" + name, attrs)
            node.text = value
            return node

        root = ET.Element("{" + namespace + "}Task", version="1.4")
        child(child(root, "RegistrationInfo"), "Description", "SpinePredict native supervisor: " + str(ROOT))
        trigger = child(child(root, "Triggers"), "LogonTrigger")
        child(trigger, "Enabled", "true")
        child(trigger, "UserId", user)
        principal = child(child(root, "Principals"), "Principal", id="Author")
        child(principal, "UserId", user)
        child(principal, "LogonType", "InteractiveToken")
        child(principal, "RunLevel", "LeastPrivilege")
        settings = child(root, "Settings")
        for key, value in {"MultipleInstancesPolicy": "IgnoreNew", "DisallowStartIfOnBatteries": "false",
                           "StopIfGoingOnBatteries": "false", "StartWhenAvailable": "true", "Enabled": "true",
                           "ExecutionTimeLimit": "PT0S"}.items():
            child(settings, key, value)
        restart = child(settings, "RestartOnFailure")
        child(restart, "Interval", "PT1M")
        child(restart, "Count", "999")
        action = child(child(root, "Actions", Context="Author"), "Exec")
        child(action, "Command", config.python)
        child(action, "Arguments", subprocess.list2cmdline([str(WEB / "deploy.py"), "_supervise", "--binary", binary]))
        child(action, "WorkingDirectory", str(ROOT))
        xml = STATE / "autostart.xml"
        atomic_write(xml, ET.tostring(root, encoding="utf-16", xml_declaration=True))
        checked(["schtasks", "/Create", "/TN", name, "/XML", str(xml)])
        write_json("autostart.json", {"platform": "windows", "name": name})
    elif sys.platform == "darwin":
        label = "org.spinepredict." + identity
        path = Path.home() / "Library" / "LaunchAgents" / (label + ".plist")
        if path.exists():
            raise Pending("LaunchAgent đã tồn tại; không ghi đè file không được quản lý.")
        confirm("Cài macOS LaunchAgent cho user hiện tại, chạy sau đăng nhập và phục hồi supervisor khi crash?")
        value = {"Label": label, "ProgramArguments": [config.python, str(WEB / "deploy.py"), "_supervise", "--binary", binary],
                 "WorkingDirectory": str(ROOT), "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False},
                 "ThrottleInterval": 15}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps(value))
        path.chmod(0o600)
        # Persist before bootstrap, so a rejected bootstrap is recoverable by uninstall.
        write_json("autostart.json", {"platform": "macos", "label": label, "path": str(path)})
        checked(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    else:
        raise Pending("Auto-start hỗ trợ Windows/macOS; nền tảng này chưa được hỗ trợ.")
    print("Đã cài auto-start; không có API setup/DNS trong tác vụ khởi động. Kiểm tra start/status/verify.")


def launch_registered():
    cached = read_json("autostart.json")
    if not cached:
        return False
    if cached.get("platform") == "windows" and os.name == "nt":
        checked(["schtasks", "/Run", "/TN", cached["name"]])
    elif cached.get("platform") == "macos" and sys.platform == "darwin":
        checked(["launchctl", "kickstart", f"gui/{os.getuid()}/{cached['label']}"])
    else:
        raise Pending("Auto-start cache thuộc OS khác; không chạy cấu hình đó trên máy này.")
    return True


def uninstall(config):
    cached = read_json("autostart.json")
    if not cached:
        print("Auto-start chưa được cài.")
        return
    confirm("Gỡ cấu hình auto-start được CLI này quản lý? Không xóa DNS/tunnel/model/dữ liệu.")
    from .processes import stop
    stop(config)
    if cached.get("platform") == "windows" and os.name == "nt":
        checked(["schtasks", "/Delete", "/TN", cached["name"], "/F"])
    elif cached.get("platform") == "macos" and sys.platform == "darwin":
        path = Path(cached["path"])
        if path.parent != Path.home() / "Library" / "LaunchAgents" or path.name != cached["label"] + ".plist":
            raise DeployError("LaunchAgent path không hợp lệ; không xóa.")
        if path.exists():
            with path.open("rb") as stream:
                value = plistlib.load(stream)
            if value.get("Label") != cached["label"] or str(WEB / "deploy.py") not in value.get("ProgramArguments", []):
                raise DeployError("LaunchAgent đã thay đổi; không xóa file khác.")
        # Already-unloaded is harmless; remove only our verified local plist.
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{cached['label']}"], capture_output=True)
        path.unlink(missing_ok=True)
    else:
        raise Pending("Auto-start cache thuộc OS khác; gỡ trên đúng máy đã cài.")
    (STATE / "autostart.json").unlink(missing_ok=True)
    print("Đã gỡ auto-start; tài nguyên remote và dữ liệu vẫn còn.")
