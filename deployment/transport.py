"""Verified HTTPS, including Python installations without a default CA bundle."""
import os
from pathlib import Path
import socket
import ssl
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen as standard_urlopen

from .common import ROOT, DeployError


def https_context(values=None):
    settings = {**os.environ, **(values or {})}
    ca_file = settings.get("SSL_CERT_FILE")
    ca_dir = settings.get("SSL_CERT_DIR")
    locations = {}
    for key, value in (("cafile", ca_file), ("capath", ca_dir)):
        if value:
            path = Path(value).expanduser()
            path = path if path.is_absolute() else ROOT / path
            if not (path.is_file() if key == "cafile" else path.is_dir()):
                raise DeployError("SSL_CERT_FILE/SSL_CERT_DIR không tồn tại hoặc sai loại; dùng CA bundle tin cậy của interpreter.")
            locations[key] = str(path)
    try:
        context = ssl.create_default_context(**locations)
        # capath loads roots lazily, so an empty get_ca_certs() alone is not
        # sufficient to conclude that the default trust store is missing.
        if (not locations and not context.get_ca_certs()
                and not ssl.get_default_verify_paths().capath):
            try:
                import certifi
            except ImportError:
                pass  # No installation or modification of the Python environment.
            else:
                context.load_verify_locations(cafile=certifi.where())
        return context
    except (OSError, ValueError) as exc:
        raise DeployError("Không đọc được CA bundle HTTPS; kiểm tra SSL_CERT_FILE/SSL_CERT_DIR hoặc certifi đã cài.") from None


def urlopen(url, *, values=None, **kwargs):
    address = url.full_url if hasattr(url, "full_url") else url
    if urlsplit(address).scheme == "https":
        kwargs["context"] = https_context(values)
    return standard_urlopen(url, **kwargs)


def network_reason(exc):
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "tls-certificate"
    if isinstance(reason, ssl.SSLError):
        return "tls-handshake"
    if isinstance(reason, socket.gaierror):
        return "dns-resolution"
    if isinstance(reason, TimeoutError):
        return "timeout"
    if isinstance(reason, ConnectionRefusedError):
        return "connection-refused"
    return "connection"
