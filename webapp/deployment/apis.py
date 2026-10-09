"""Official APIs only; errors never echo response bodies or request headers."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .common import DeployError
from .transport import https_context, network_reason, urlopen


class APIError(DeployError):
    def __init__(self, provider, status=None, reason=None):
        self.status = status
        self.reason = reason
        details = {
            "tls-certificate": "TLS certificate verification failed; kiểm tra CA của Python hoặc SSL_CERT_FILE",
            "tls-handshake": "TLS handshake failed; kiểm tra proxy và TLS của interpreter",
            "dns-resolution": "DNS resolution failed; kiểm tra Internet/DNS/proxy",
            "timeout": "connection/read timeout; kiểm tra Internet/proxy rồi thử lại",
            "connection-refused": "connection refused; kiểm tra network/proxy",
            "connection": "network connection failed; kiểm tra Internet/proxy",
            "invalid-response": "response không phải JSON hợp lệ; kiểm tra proxy/API",
            "api-response": "API trả success=false/response không hợp lệ; kiểm tra quyền/token",
        }
        detail = f"HTTP {status}" if status else details.get(reason, "network/API chưa xác minh")
        super().__init__(f"{provider} API: {detail}. Kiểm tra quyền/token/network; không tự xóa hoặc tạo lại tài nguyên.")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Never forward API authentication to another URL.


class API:
    def __init__(self, base, provider, headers, values=None):
        self.base, self.provider, self.headers = base, provider, headers
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=https_context(values)))

    def request(self, method, path, body=None, query=None):
        url = self.base + path + (("?" + urlencode(query)) if query else "")
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/json", "Content-Type": "application/json", **self.headers}
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                with self.opener.open(Request(url, data=data, headers=headers, method=method), timeout=20) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise APIError(self.provider)
                    return json.loads(raw) if raw else {}
            except HTTPError as exc:
                if method == "GET" and (exc.code == 429 or exc.code >= 500) and attempt + 1 < attempts:
                    time.sleep(1 + attempt)
                    continue
                raise APIError(self.provider, exc.code) from None
            except (URLError, OSError) as exc:
                reason = network_reason(exc)
                if method == "GET" and reason not in {"tls-certificate", "tls-handshake"} and attempt + 1 < attempts:
                    time.sleep(1 + attempt)
                    continue
                raise APIError(self.provider, reason=reason) from None
            except ValueError:
                raise APIError(self.provider, reason="invalid-response") from None


class Cloudflare(API):
    def __init__(self, config):
        config.require_api()
        super().__init__("https://api.cloudflare.com/client/v4", "Cloudflare", {
            "Authorization": "Bearer " + config.values["CLOUDFLARE_API_TOKEN"]
        }, config.values)
        self.account, self.zone = config.account, config.zone
        self.tunnels = f"/accounts/{self.account}/cfd_tunnel"

    def result(self, method, path, body=None, query=None):
        response = self.request(method, path, body, query)
        if response.get("success") is not True:
            raise APIError("Cloudflare", reason="api-response")
        return response.get("result")

    def pages(self, path, query=None):
        items = []
        for page in range(1, 1001):
            response = self.request("GET", path, query={**(query or {}), "page": page, "per_page": 100})
            if response.get("success") is not True or not isinstance(response.get("result"), list):
                raise APIError("Cloudflare", reason="api-response")
            values = response["result"]
            items.extend(values)
            info = response.get("result_info", {})
            if page >= info.get("total_pages", 1001) or len(values) < 100:
                return items
        raise DeployError("API pagination vượt giới hạn; không chọn tài nguyên không đầy đủ.")

    def zone_info(self):
        return self.result("GET", f"/zones/{self.zone}")

    def find_tunnels(self):
        return [t for t in self.pages(self.tunnels, {"name": "spinepredict-tunnel", "is_deleted": "false"})
                if t.get("name") == "spinepredict-tunnel" and not t.get("deleted_at")]

    def tunnel(self, tunnel_id):
        return self.result("GET", self.tunnels + "/" + quote(tunnel_id, safe=""))

    def create_tunnel(self):
        return self.result("POST", self.tunnels, {"name": "spinepredict-tunnel", "config_src": "cloudflare"})

    def token(self, tunnel_id):
        return self.result("GET", self.tunnels + f"/{tunnel_id}/token")

    def connections(self, tunnel_id):
        # The deprecated tunnel.connections field is not used for verification.
        return self.result("GET", self.tunnels + f"/{tunnel_id}/connections")

    def tunnel_config(self, tunnel_id):
        try:
            result = self.result("GET", self.tunnels + f"/{tunnel_id}/configurations")
        except APIError as exc:
            if exc.status != 404:
                raise
            return {}
        return result.get("config") or {}

    def set_tunnel_config(self, tunnel_id, config):
        return self.result("PUT", self.tunnels + f"/{tunnel_id}/configurations", {"config": config})

    def dns(self, domain=None):
        return self.pages(f"/zones/{self.zone}/dns_records", {"name": domain} if domain else None)

    def set_dns(self, body, record_id=None):
        path = f"/zones/{self.zone}/dns_records"
        return self.result("PATCH" if record_id else "POST", path + ("/" + record_id if record_id else ""), body)

    def replace_address_records(self, records, body):
        # Single database transaction; no sequential DELETE followed by POST.
        return self.result("POST", f"/zones/{self.zone}/dns_records/batch", {
            "deletes": [{"id": record["id"]} for record in records], "posts": [body]
        })


class Spaceship(API):
    def __init__(self, config):
        key, secret = config.values.get("SPACESHIP_API_KEY"), config.values.get("SPACESHIP_API_SECRET")
        if not key or not secret:
            raise DeployError("Spaceship credentials chưa có; cập nhật nameserver thủ công nếu cần.")
        super().__init__("https://spaceship.dev/api/v1", "Spaceship", {"X-API-Key": key, "X-API-Secret": secret}, config.values)
        self.domain_path = "/domains/" + quote(config.domain, safe="")

    def domain_info(self):
        return self.request("GET", self.domain_path)

    def set_nameservers(self, hosts):
        return self.request("PUT", self.domain_path + "/nameservers", {"provider": "custom", "hosts": hosts})


def dns_query(domain, kind, values=None):
    """Public recursive DNS read. DS lookup is a fallback, not a registrar DNSSEC API."""
    try:
        request = Request("https://cloudflare-dns.com/dns-query?" + urlencode({"name": domain, "type": kind}),
                          headers={"Accept": "application/dns-json"})
        with urlopen(request, timeout=10, values=values) as response:
            body = json.load(response)
        if body.get("Status") != 0:
            return None
        record_type = {"NS": 2, "DS": 43}[kind]
        return [answer["data"].lower().rstrip(".") for answer in body.get("Answer", []) if answer.get("type") == record_type]
    except (OSError, ValueError):
        return None
