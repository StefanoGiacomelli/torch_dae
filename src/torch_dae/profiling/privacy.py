"""Privacy-safe provenance validator (Section 6 / docs/profiling/technical-cards.md).

Hardware/software/execution-context evidence must describe a configuration class, never a
physical machine. This validator rejects obvious prohibited fields or path leakage anywhere in
a serialized Technical Card payload.
"""

from __future__ import annotations

import getpass
import os
import re
import socket
from collections.abc import Mapping, Sequence

_PROHIBITED_KEY_SUBSTRINGS = (
    "hostname",
    "host_name",
    "username",
    "user_name",
    "ip_address",
    "mac_address",
    "machine_uuid",
    "machine_id",
    "serial_number",
    "serial",
    "password",
    "credential",
    "token",
    "home_dir",
    "homedir",
)

_MAC_ADDRESS_PATTERN = re.compile(r"\b([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
_IPV4_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _local_secrets() -> tuple[str, ...]:
    secrets = []
    try:
        secrets.append(socket.gethostname())
    except OSError:  # pragma: no cover - defensive
        pass
    try:
        secrets.append(getpass.getuser())
    except (OSError, KeyError):  # pragma: no cover - defensive
        pass
    home = os.path.expanduser("~")
    if home and home != "~":
        secrets.append(home)
    return tuple(value for value in secrets if value)


class PrivacyViolation(ValueError):
    """Raised when a candidate Technical Card payload leaks prohibited machine identity."""


def validate_privacy_safe(payload: object, *, path: str = "$") -> None:
    """Recursively reject prohibited keys, patterns, and local machine-identity leakage.

    Raises
    ------
    PrivacyViolation
        If a prohibited key name, MAC address, IPv4 literal, or the current process's own
        hostname/username/home directory appears anywhere in the payload.
    """

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(bad in lowered for bad in _PROHIBITED_KEY_SUBSTRINGS):
                raise PrivacyViolation(f"prohibited field at {path}.{key}")
            validate_privacy_safe(value, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        for index, item in enumerate(payload):
            validate_privacy_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(payload, str):
        _validate_string(payload, path=path)


def _validate_string(value: str, *, path: str) -> None:
    if _MAC_ADDRESS_PATTERN.search(value):
        raise PrivacyViolation(f"apparent MAC address leaked at {path}")
    if _IPV4_PATTERN.search(value):
        raise PrivacyViolation(f"apparent IPv4 address leaked at {path}")
    for secret in _local_secrets():
        if len(secret) >= 3 and secret in value:
            raise PrivacyViolation(f"local machine-identity string leaked at {path}")
