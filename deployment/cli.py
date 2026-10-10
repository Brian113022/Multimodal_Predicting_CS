import argparse
import sys

from .common import Config, DeployError, Pending


def main(argv=None):
    # Windows redirected output may default to cp1252, which cannot print
    # Vietnamese diagnostics (including argparse help).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Native Cloudflare Named Tunnel deployment (Windows/macOS)")
    parser.add_argument("action", nargs="?", default="run", choices=(
        "run", "setup", "start", "stop", "restart", "status", "verify",
        "autostart-install", "autostart-uninstall", "_supervise"
    ))
    parser.add_argument("--binary", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-token", action="store_true", help="setup: lấy lại token cho cùng tunnel, không tạo tunnel mới")
    args = parser.parse_args(argv)
    if args.refresh_token and args.action != "setup":
        parser.error("--refresh-token chỉ dùng với setup")
    try:
        config = Config.load()
        if args.action == "setup":
            from .setup import setup
            setup(config, refresh_token=args.refresh_token)
        elif args.action in {"run", "start"}:
            from .setup import require_state, setup
            from .processes import cloudflared, port_open, preflight, start, stop
            if args.action == "run":
                # Fail local prerequisites before creating remote resources.
                cloudflared(config)
                preflight(config)
                try:
                    require_state(config)
                except Pending:
                    setup(config)
                stop(config)
                if port_open(config.port):
                    raise DeployError("Port còn bị app ngoài deployment chiếm; chưa khởi chạy app mới. Dừng app đó trước.")
            start(config, preflight_done=args.action == "run")
            from .verify import verify
            return verify(config)
        elif args.action == "stop":
            from .processes import stop
            stop(config)
        elif args.action == "restart":
            from .processes import start, stop
            from .setup import require_state
            before = require_state(config)["tunnel_id"]
            stop(config)
            start(config)
            if require_state(config)["tunnel_id"] != before:
                raise DeployError("Tunnel identity đổi ngoài dự kiến; kiểm tra state.")
            print("PASS: restart tái sử dụng tunnel ID và domain local; verify kiểm tra DNS/HTTPS remote.")
        elif args.action == "status":
            from .processes import status
            status(config)
        elif args.action == "verify":
            from .verify import verify
            return verify(config)
        elif args.action == "autostart-install":
            from .autostart import install
            install(config)
        elif args.action == "autostart-uninstall":
            from .autostart import uninstall
            uninstall(config)
        elif args.action == "_supervise":
            from .processes import cloudflared, supervise
            supervise(config, args.binary or cloudflared(config))
        return 0
    except Pending as exc:
        print("PENDING:", str(exc))
        return 2
    except DeployError as exc:
        print("ERROR:", str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Đã ngắt CLI. Process chạy background vẫn được quản lý; dùng stop nếu cần.")
        return 130
    except Exception:
        print("ERROR: deployment gặp lỗi chưa dự kiến; không in dữ liệu API/credentials. Kiểm tra cấu hình và logs local.", file=sys.stderr)
        return 1
