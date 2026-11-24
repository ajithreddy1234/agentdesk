"""Signed bearer tokens with roles (customer / supervisor).

Tokens are `base64url(payload).hmac_sha256` signed with AUTH_SECRET and carry an expiry.
The demo issues them from simple logins (a shared demo customer password, and supervisor
accounts from SUPERVISORS="name:password,..."). In production, replace `login_*` with your
identity provider and keep `require_customer` / `require_supervisor` as the gate.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import Header, HTTPException

AUTH_SECRET = os.getenv("AUTH_SECRET") or secrets.token_hex(32)
TOKEN_TTL_S = int(os.getenv("TOKEN_TTL_S", str(8 * 3600)))
DEMO_CUSTOMER_PASSWORD = os.getenv("DEMO_CUSTOMER_PASSWORD", "demo-customer")


def _supervisors() -> dict[str, str]:
    raw = os.getenv("SUPERVISORS", "supervisor:demo-supervisor")
    out = {}
    for pair in raw.split(","):
        name, _, pw = pair.strip().partition(":")
        if name and pw:
            out[name] = pw
    return out


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(role: str, subject: str | int) -> str:
    payload = _b64(json.dumps({"r": role, "s": str(subject), "e": int(time.time()) + TOKEN_TTL_S}).encode())
    sig = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify(token: str) -> dict:
    payload, _, sig = (token or "").partition(".")
    good = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not payload or not hmac.compare_digest(sig, good):
        raise HTTPException(401, "Invalid token")
    data = json.loads(_unb64(payload))
    if data.get("e", 0) < time.time():
        raise HTTPException(401, "Token expired")
    return data


def _bearer(authorization: str | None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    return verify(authorization.split(" ", 1)[1].strip())


def require_customer(authorization: str | None = Header(None)) -> int:
    data = _bearer(authorization)
    if data.get("r") != "customer":
        raise HTTPException(403, "Customer token required")
    return int(data["s"])


def require_supervisor(authorization: str | None = Header(None)) -> str:
    data = _bearer(authorization)
    if data.get("r") != "supervisor":
        raise HTTPException(403, "Supervisor token required")
    return data["s"]


def check_customer_login(password: str) -> bool:
    return hmac.compare_digest(password or "", DEMO_CUSTOMER_PASSWORD)


def check_supervisor_login(username: str, password: str) -> bool:
    expected = _supervisors().get(username or "")
    return expected is not None and hmac.compare_digest(password or "", expected)
