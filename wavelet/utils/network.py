from __future__ import annotations

import socket


def get_free_port(start: int = 29500, end: int = 65535) -> int:
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port found")
