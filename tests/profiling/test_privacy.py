from __future__ import annotations

import pytest

from torch_dae.profiling.privacy import PrivacyViolation, validate_privacy_safe


def test_clean_payload_passes() -> None:
    validate_privacy_safe(
        {
            "cpu_model": "Apple M4 Pro",
            "cpu_architecture": "arm64",
            "physical_cores": 10,
            "nested": {"os_name": "macOS", "values": [1, 2, 3]},
        }
    )


@pytest.mark.parametrize(
    "key",
    [
        "hostname",
        "host_name",
        "username",
        "user_name",
        "ip_address",
        "mac_address",
        "machine_uuid",
        "machine_id",
        "serial_number",
        "password",
        "credential",
        "api_token",
        "home_dir",
    ],
)
def test_rejects_prohibited_keys(key: str) -> None:
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe({key: "some-value"})


def test_rejects_mac_address_pattern() -> None:
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe({"note": "device at 00:1B:44:11:3A:B7"})


def test_rejects_ipv4_pattern() -> None:
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe({"note": "reachable at 192.168.1.42"})


def test_rejects_current_hostname_leak() -> None:
    import socket

    hostname = socket.gethostname()
    if len(hostname) < 3:
        pytest.skip("hostname too short to be a meaningful leak signal")
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe({"note": f"observed on {hostname}"})


def test_rejects_home_directory_leak() -> None:
    import os

    home = os.path.expanduser("~")
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe({"path": f"{home}/some/file.json"})


def test_recurses_into_lists() -> None:
    with pytest.raises(PrivacyViolation):
        validate_privacy_safe([{"ok": 1}, {"password": "x"}])
