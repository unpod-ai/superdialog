"""SSRF guard for tool URLs, validated AFTER template rendering.

Tool URLs are Jinja templates over {slots, env, results}; the target host can
arrive via ``{{ env.X }}`` or ``{{ slots.Y }}`` interpolation, and playbooks are
optimizer- or user-generated. So the guard runs on the RENDERED url (not the
template), mirroring the SSTI posture the sandboxed renderer already takes.

Folds every numeric IPv4 spelling ``getaddrinfo`` honours (decimal / octal /
hex / short) plus IPv4-mapped IPv6 to a concrete address before range-checking,
so ``2130706433`` / ``0177.0.0.1`` / ``::ffff:169.254.169.254`` cannot slip past
the private-range block the way a canonical-only check would.

Best-effort: DNS rebinding can bypass a validation-time check because a genuine
hostname is resolved at request time, not here (a synchronous lookup would stall
the event loop). The real protection remains that tool URLs are configuration,
not caller-supplied free text.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# Hostnames that must never be targeted even when they are not literal IPs
# (DNS-based SSRF to cloud-metadata endpoints or loopback aliases).
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata",
    }
)


def resolve_literal_ip(host: str) -> IPAddress | None:
    """Return the IP literal *host* denotes, or ``None`` when *host* is a name.

    Canonical forms go through :mod:`ipaddress`; non-canonical numeric IPv4
    (decimal / octal / hex / short — ``2130706433``, ``0x7f.0.0.1``,
    ``0177.0.0.1``, ``127.1``) fold via :func:`socket.inet_aton`, which the C
    resolver honours. A genuine hostname returns ``None`` (resolved at request
    time, not here).
    """
    host = host.strip("[]")
    if not host:
        return None
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def is_internal_ip(addr: IPAddress) -> bool:
    """True when *addr* (or the IPv4 an IPv4-mapped IPv6 embeds) is private,
    loopback, link-local, reserved, or unspecified.

    Unwrapping ``ipv4_mapped`` matters on Python < 3.13, where the range flags
    on the IPv6 wrapper do not reflect the embedded IPv4.
    """
    candidates: list[IPAddress] = [addr]
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        candidates.append(mapped)
    return any(
        a.is_private
        or a.is_loopback
        or a.is_link_local
        or a.is_reserved
        or a.is_unspecified
        for a in candidates
    )


def validate_url(url: str, *, allow_private_hosts: bool = False) -> None:
    """Raise ``ValueError`` if *url* is an SSRF risk.

    Always: scheme must be http/https and a hostname must be present. Unless
    ``allow_private_hosts`` (local-dev opt-in): blocked-hostname aliases and
    literal private/loopback/link-local/reserved IPs — across every numeric
    spelling — are rejected.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http"):
        raise ValueError(f"blocked tool URL: invalid scheme {parts.scheme!r}")
    hostname = parts.hostname or ""
    if not hostname:
        raise ValueError("blocked tool URL: missing hostname")
    if allow_private_hosts:
        return
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError(f"blocked tool URL: blocked hostname {hostname!r}")
    addr = resolve_literal_ip(hostname)
    if addr is not None and is_internal_ip(addr):
        raise ValueError(f"blocked tool URL: private/reserved address {hostname!r}")
