from urllib.parse import urlparse

from .apis import Cloudflare, dns_query
from .common import DeployError, read_json
from .processes import alive, app_health, get_json, readiness
from .setup import validate_zone


def verify(config):
    results = []

    def report(label, status):
        results.append(status)
        print(status + ": " + label)

    report("Python localhost:" + str(config.port) + " và model", "PASS" if app_health(config) else "FAIL")
    state = read_json("setup.json")
    try:
        cf = Cloudflare(config)
        zone = cf.zone_info()
        expected = validate_zone(config, zone)
        report("Domain/zone/account Cloudflare khớp", "PASS")
        zone_active = zone.get("status") == "active"
        report("Cloudflare Zone Active", "PASS" if zone_active else "PENDING")
        public_ns = dns_query(config.domain, "NS", config.values)
        report("Nameservers public trỏ Cloudflare", "PASS" if public_ns is not None and set(public_ns) == {s.lower().rstrip('.') for s in expected} else "PENDING")
        tunnel_id = state.get("tunnel_id")
        if not tunnel_id:
            report("Named Tunnel đã setup", "PENDING")
        else:
            tunnel = cf.tunnel(tunnel_id)
            report("Named Tunnel remotely managed đúng tên", "PASS" if tunnel.get("config_src") == "cloudflare" and tunnel.get("name") == "spinepredict-tunnel" else "FAIL")
            records = cf.dns(config.domain)
            matches = [r for r in records if r.get("type") == "CNAME" and r.get("content", "").rstrip(".") == tunnel_id + ".cfargotunnel.com" and r.get("proxied") is True]
            report("DNS CNAME proxied trỏ đúng tunnel", "PASS" if len(matches) == 1 else "FAIL")
            rules = cf.tunnel_config(tunnel_id).get("ingress", [])
            report("Published route tới localhost", "PASS" if any(r.get("hostname") == config.domain and r.get("service") == config.origin and not r.get("path") for r in rules) else "FAIL")
            connections = cf.connections(tunnel_id) or []
            registered = any(client.get("conns") for client in connections)
            report("Named Tunnel có connector đăng ký kết nối", "PASS" if registered else "PENDING")
    except DeployError as exc:
        report("Remote API chưa xác minh: " + str(exc), "PENDING")
        zone_active = False
    diagnostics = {}
    status, body, final_url = get_json(
        "https://" + config.domain + "/api/health", timeout=10, values=config.values,
        headers={"User-Agent": "SpinePredictDeployment/1.0", "Accept": "application/json"},
        diagnostics=diagnostics,
    )
    public_healthy = status == 200 and body.get("model_loaded") is True and body.get("model", {}).get("bundle_type") == "fusion_subsidence_full_late_fusion"
    public_healthy = public_healthy and urlparse(final_url).scheme == "https" and urlparse(final_url).hostname == config.domain
    public_label = "Public HTTPS certificate và model health"
    if not public_healthy:
        if status == 403:
            detail = "HTTP 403; kiểm tra Cloudflare Security Events/Access và quyền truy cập"
            if diagnostics.get("cf_ray"):
                detail += "; Ray ID: " + diagnostics["cf_ray"]
        elif status is not None:
            detail = f"HTTP {status}; payload/redirect chưa xác minh đúng app"
        else:
            detail = {
                "tls-certificate": "TLS certificate verification failed; kiểm tra CA/edge certificate",
                "tls-handshake": "TLS handshake failed",
                "dns-resolution": "DNS resolution failed; resolver có thể chưa cập nhật",
                "timeout": "HTTPS connection/read timeout",
                "invalid-response": "response chưa đọc được JSON health",
            }.get(diagnostics.get("reason"), "chưa kết nối được HTTPS")
        public_label += " (" + detail + ")"
    report(public_label, "PASS" if public_healthy else "PENDING")
    runtime = read_json("runtime.json")
    if alive(runtime) and runtime.get("metrics_port"):
        report("Connector readiness local", "PASS" if readiness(runtime["metrics_port"]) else "PENDING")
    print("Kiểm tra hoàn tất.")
    return 1 if "FAIL" in results else 2 if "PENDING" in results else 0
