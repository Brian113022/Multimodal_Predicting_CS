"""Automatic, idempotent setup with validated DNS/delegation changes."""
from copy import deepcopy
import fnmatch
import hashlib
import json
import uuid

from .apis import APIError, Cloudflare, Spaceship, dns_query
from .common import DeployError, FileLock, Pending, confirm, load_token, read_json, save_token, write_json


def valid_state(config, state, complete=True):
    return (state.get("version") == 1 and state.get("binding") == config.binding()
            and bool(state.get("tunnel_id")) and (not complete or state.get("complete") is True))


def require_state(config):
    state = read_json("setup.json")
    if not valid_state(config, state):
        raise Pending("Setup chưa hoàn tất hoặc cấu hình đã đổi; chạy deploy.py setup.")
    load_token(state["tunnel_id"])
    return state


def validate_zone(config, zone):
    if zone.get("name", "").lower() != config.domain or zone.get("account", {}).get("id") != config.account:
        raise DeployError("Zone ID không khớp DOMAIN/account; chưa thay đổi tài nguyên.")
    hosts = zone.get("name_servers") or []
    if len(hosts) < 2:
        raise DeployError("Cloudflare chưa trả nameservers hợp lệ cho zone.")
    return hosts


def nameservers(config, zone, cf):
    expected = validate_zone(config, zone)
    normal = lambda hosts: {host.lower().rstrip(".") for host in hosts}
    try:
        registrar = Spaceship(config)
        info = registrar.domain_info()
        current = info.get("nameservers", {}).get("hosts", [])
        if not current:
            raise DeployError("Spaceship không trả nameserver hiện tại.")
    except DeployError:
        current = dns_query(config.domain, "NS", config.values)
        if current is not None and normal(current) == normal(expected):
            print("PASS: recursive DNS hiện trỏ về Cloudflare; Spaceship API chưa xác minh, không cần đổi nameservers.")
            return True
        print("PENDING: Spaceship API không đọc được domain (credentials/quyền/API).")
        print("Cập nhật thủ công tại Spaceship → Domain → Nameservers → Custom:", ", ".join(expected))
        return False
    if normal(current) == normal(expected):
        print("PASS: nameservers registrar đã đúng; không gửi PUT.")
        return True
    records = cf.dns()
    print("Cloudflare hiện có", len(records), "DNS records. Không tự import/xóa MX/TXT hoặc records khác.")
    ds = dns_query(config.domain, "DS", config.values)
    if ds:
        print("PENDING: domain còn DS/DNSSEC. Gỡ DS cũ theo quy trình registrar trước khi đổi nameservers.")
        return False
    print("DNSSEC registrar API: chưa được Spaceship công bố; DS recursive", "không thấy" if ds == [] else "chưa xác minh")
    confirm("Đổi nameservers cho " + config.domain + "\nHiện tại: " + ", ".join(current)
            + "\nDự kiến: " + ", ".join(expected)
            + "\nGiữ nguyên các DNS records khác; đã kiểm tra DS trước khi đổi nameservers.")
    # Re-read just before writing to avoid overwriting a concurrent registrar edit.
    if normal(registrar.domain_info().get("nameservers", {}).get("hosts", [])) != normal(current):
        raise Pending("Nameservers vừa thay đổi ở nơi khác; chạy setup lại để xem dự kiến mới.")
    try:
        registrar.set_nameservers(expected)
    except APIError:
        print("PENDING: cập nhật nameserver bị từ chối/timeout; kiểm tra lại trước khi retry.")
        print("Spaceship → Nameservers → Custom:", ", ".join(expected))
        return False
    print("PENDING: nameservers đã gửi; DNS/zone cần propagation, dùng verify để kiểm tra lại.")
    return True


def ingress_plan(config, existing):
    desired = deepcopy(existing)
    rules = desired.setdefault("ingress", [])
    if not isinstance(rules, list):
        raise DeployError("Tunnel ingress không hợp lệ; giữ nguyên để kiểm tra thủ công.")
    matches = [rule for rule in rules if rule.get("hostname", "").lower() == config.domain and not rule.get("path")]
    if len(matches) > 1:
        raise Pending("Tunnel có nhiều route toàn domain; cần xử lý thủ công, không tự xóa route.")
    if matches:
        # Preserve originRequest/access and every unrelated rule/settings.
        matches[0]["service"] = config.origin
    else:
        if any(rule.get("hostname") and fnmatch.fnmatchcase(config.domain, rule["hostname"])
               for rule in rules):
            raise Pending("Route wildcard/path hiện có bao phủ domain; kiểm tra auth/route thủ công trước khi thêm route.")
        rules.insert(0, {"hostname": config.domain, "service": config.origin})
    if not any(not rule.get("hostname") and not rule.get("path") for rule in rules):
        rules.append({"service": "http_status:404"})
    return desired


def approve_change(name, config, tunnel_id, before, after, message):
    """Remember approval only for this exact resource, snapshot and desired change."""
    plan = {"binding": config.binding(), "tunnel_id": tunnel_id, "before": before, "after": after}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    filename = name + "-approval.json"
    if read_json(filename).get("fingerprint") == fingerprint:
        print("Tái sử dụng xác nhận đã lưu cho đúng thay đổi " + name + ".")
        return
    confirm(message)
    write_json(filename, {"fingerprint": fingerprint})


def ensure_ingress(config, cf, tunnel_id):
    existing = cf.tunnel_config(tunnel_id)
    desired = ingress_plan(config, existing)
    if desired == existing:
        print("PASS: tunnel route đã đúng; không gửi PUT.")
        return
    rules = existing.get("ingress", [])
    replacing = any(rule.get("hostname", "").lower() == config.domain for rule in rules)
    safe_catchall = all(rule.get("hostname") or rule.get("service") == "http_status:404" for rule in rules)
    if replacing or not safe_catchall:
        approve_change("ingress", config, tunnel_id, existing, desired,
                       f"Cập nhật Published Application Route: {config.domain} → {config.origin}. Giữ nguyên routes/auth khác.")
    else:
        print(f"Tự động thêm Published Application Route: {config.domain} → {config.origin}.")
    latest = cf.tunnel_config(tunnel_id)
    if latest == desired:
        print("PASS: route đã được lưu; không gửi lại PUT.")
        return
    if latest != existing:
        raise Pending("Tunnel config vừa thay đổi; chạy setup lại, chưa ghi đè.")
    write_json("ingress-backup.json", existing)
    try:
        cf.set_tunnel_config(tunnel_id, desired)
    except APIError as exc:
        # A timed-out PUT may already be committed. Read back, never retry a write blindly.
        print("Route API chưa xác minh; đang đọc lại cấu hình Cloudflare.")
        try:
            applied = cf.tunnel_config(tunnel_id) == desired
        except DeployError:
            applied = False
        if applied:
            print("PASS: Cloudflare đã lưu route dù request trước gặp lỗi; tiếp tục DNS.")
            return
        if exc.status and 400 <= exc.status < 500 and exc.status not in {408, 429}:
            raise
        raise Pending(f"Route chưa xác minh sau lỗi: {exc} Chạy lại run.sh để đọc/reconcile cùng tunnel; chưa gửi lại PUT.") from None


def matching_dns(records, target):
    address = [r for r in records if r.get("type") in {"CNAME", "A", "AAAA"}]
    if len(address) == 1:
        record = address[0]
        if (record["type"] == "CNAME" and record.get("content", "").lower().rstrip(".") == target
                and record.get("proxied") is True):
            return record
    return None


def ensure_dns(config, cf, tunnel_id):
    records = cf.dns(config.domain)
    address = [r for r in records if r.get("type") in {"CNAME", "A", "AAAA"}]
    target = tunnel_id + ".cfargotunnel.com"
    old = address[0] if len(address) == 1 else None
    matched = matching_dns(records, target)
    if matched:
        print("PASS: DNS CNAME/proxy đã đúng; không gửi POST/PATCH.")
        return matched["id"]
    body = {"type": "CNAME", "name": config.domain, "content": target, "proxied": True, "ttl": 1}
    if len(address) > 1:
        operation = "Thay đúng các apex address records trong một API batch: " + ", ".join(
            record["id"] + " (" + record["type"] + ", " + record.get("content", "") + ")" for record in address
        )
    else:
        operation = "Tạo" if old is None else "Cập nhật record " + old["id"] + " (" + old["type"] + ")"
    if address:
        approve_change("dns", config, tunnel_id, records, body,
                       f"{operation} DNS: {config.domain} CNAME {target}, proxied=true. Không thay MX/TXT/records khác.")
    else:
        print(f"Tự động tạo DNS: {config.domain} CNAME {target}, proxied=true.")
    latest = cf.dns(config.domain)
    matched = matching_dns(latest, target)
    if matched:
        print("PASS: DNS đã trỏ đúng tunnel; không gửi lại thay đổi.")
        return matched["id"]
    if latest != records:
        raise Pending("DNS vừa thay đổi ở nơi khác; chạy setup lại để xem thay đổi mới.")
    write_json("dns-backup.json", {"records": records})
    try:
        if len(address) > 1:
            cf.replace_address_records(address, body)
        else:
            cf.set_dns(body, old["id"] if old else None)
    except APIError as exc:
        print("DNS API chưa xác minh; đang đọc lại record Cloudflare.")
        try:
            matched = matching_dns(cf.dns(config.domain), target)
        except DeployError:
            matched = None
        if matched:
            print("PASS: DNS đã được lưu dù request trước gặp lỗi; không tạo record trùng.")
            return matched["id"]
        if exc.status and 400 <= exc.status < 500 and exc.status not in {408, 429}:
            raise
        raise Pending(f"DNS chưa xác minh sau lỗi: {exc} Chạy lại run.sh; tái sử dụng xác nhận nếu records vẫn khớp, không tự retry ghi.") from None
    try:
        matched = matching_dns(cf.dns(config.domain), target)
    except DeployError as exc:
        raise Pending(f"DNS API đã xác nhận ghi nhưng đọc lại chưa thành công: {exc} Chạy lại run.sh để kiểm tra cùng record.") from None
    if not matched:
        raise Pending("DNS đã gửi nhưng chưa đọc lại được CNAME mong muốn; chạy lại run.sh, không tự retry ghi.")
    print("PASS: DNS CNAME proxied đã trỏ tới tunnel.")
    return matched["id"]


def setup(config, refresh_token=False):
    config.require_api()
    with FileLock("setup.lock"):
        cf = Cloudflare(config)
        zone = cf.zone_info()
        validate_zone(config, zone)
        state = read_json("setup.json")
        if state and state.get("binding") != config.binding():
            raise Pending("State thuộc domain/account/port khác. Kiểm tra state; không tự thay tài nguyên.")
        print("Cloudflare zone:", zone.get("status", "unknown"))
        ns_ready = nameservers(config, zone, cf)
        tunnels = cf.find_tunnels()
        if len(tunnels) > 1:
            raise Pending("Có nhiều tunnel cùng tên spinepredict-tunnel; cần chọn/xử lý thủ công.")
        if state.get("tunnel_id"):
            tunnel = cf.tunnel(state["tunnel_id"])
            if not tunnels or tunnel.get("id") != tunnels[0].get("id"):
                raise Pending("Tunnel cache không khớp danh sách remote; không tự tạo tunnel thay thế.")
        elif tunnels:
            tunnel = tunnels[0]
        else:
            print("Tạo remotely managed Named Tunnel spinepredict-tunnel.")
            try:
                tunnel = cf.create_tunnel()
            except APIError:
                # POST may have succeeded before a timeout. Reconcile with a GET,
                # never blindly retry the creation request.
                tunnels = cf.find_tunnels()
                if len(tunnels) != 1:
                    raise
                tunnel = tunnels[0]
        if tunnel.get("config_src") != "cloudflare" or tunnel.get("deleted_at"):
            raise Pending("Tunnel hiện có không remotely managed/đã xóa; không tự đổi hoặc tạo lại.")
        try:
            tunnel_id = str(uuid.UUID(tunnel["id"]))
        except (KeyError, ValueError):
            raise DeployError("Cloudflare trả tunnel ID không hợp lệ.") from None
        state.update(version=1, binding=config.binding(), tunnel_id=tunnel_id, complete=False)
        write_json("setup.json", state)  # Persist identity before route/DNS mutations.
        print("Tunnel DNS target:", tunnel_id + ".cfargotunnel.com")
        print("Public URL dự kiến: https://" + config.domain)
        try:
            if refresh_token:
                raise Pending("Refresh requested")
            load_token(tunnel_id)
        except Pending:
            save_token(cf.token(tunnel_id), tunnel_id)
        ensure_ingress(config, cf, tunnel_id)
        state["dns_record_id"] = ensure_dns(config, cf, tunnel_id)
        state["complete"] = ns_ready
        state["name_servers"] = zone["name_servers"]
        write_json("setup.json", state)
        if not ns_ready:
            raise Pending("Tunnel/DNS đã lưu; hoàn tất nameservers/DNSSEC thủ công rồi chạy setup lại để tái sử dụng.")
        if zone.get("status") == "active":
            print("Setup đã lưu; Cloudflare zone Active. Tiếp tục khởi động app/tunnel.")
        else:
            print("Setup đã lưu; zone chưa Active (PENDING propagation). Start tái sử dụng tunnel/DNS.")
        return state
