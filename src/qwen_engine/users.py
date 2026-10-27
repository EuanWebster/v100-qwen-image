"""STUB of the Railway user store (contract C2) — replace with D2's module.

This file exists ONLY so worktree ``feature/multiuser-ui`` can be built and
tested before the server-core work lands. It implements the exact C2 surface
(``UserStore(path)`` and its nine methods) on the exact C2 file format, so the
admin routes and their tests exercise the real shape; Root reconciles it with
D2's ``users.py``, which owns the implementation, the locking and the
first-sign-in binding.

Nothing here is new contract: no extra method, no extra key, no path on the
wire. Emails are lowercased and trimmed, the file is written atomically with
mode 0600, and new users get q8 only.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

#: The one schema tag this file holds.
USERS_SCHEMA = "qwen_engine.users.v1"
#: The four permission flags, in the order the admin page shows them.
PERMISSIONS = ("q8", "2k", "edit", "official_enhancer")
#: What a newly invited user may do: Q8 only.
DEFAULT_ALLOW = {"q8": True, "2k": False, "edit": False,
                 "official_enhancer": False}
#: A cap on the store file read: this document is tiny, so anything larger is
#: a corrupt or hostile file rather than a real one.
MAX_USERS_BYTES = 1 << 20
#: Bound on how many accounts one store may hold (the file is read and
#: rewritten whole, and the admin page renders every one of them).
MAX_USERS = 500
EMAIL_MAX_CHARS = 254
DISPLAY_NAME_MAX_CHARS = 80


class UserStoreError(ValueError):
    """Typed refusal of the user store (bad email, unknown user, bad file)."""


def normalise_email(value: Any) -> str:
    """Lowercase, trimmed, bounded, and structurally an address.

    Deliberately permissive about the parts an email may contain (local-part
    punctuation, sub-domains): the gate is an allowlist, not a validator, and a
    refusal here is a typo the owner can see. What it will NOT accept is
    anything that is not a plain address — no whitespace inside, no control
    characters, one ``@``, a dot in the domain — because that string ends up in
    a JSON key and on a page.
    """
    if not isinstance(value, str):
        raise UserStoreError("email must be a string")
    text = value.strip().lower()
    if not text or len(text) > EMAIL_MAX_CHARS:
        raise UserStoreError("email must be 1-254 characters")
    if text.count("@") != 1:
        raise UserStoreError("email must contain exactly one @")
    local, _, domain = text.partition("@")
    if not local or not domain or "." not in domain:
        raise UserStoreError("email must have a local part and a dotted domain")
    for char in text:
        if char.isspace() or ord(char) < 0x20 or ord(char) == 0x7f:
            raise UserStoreError("email must not contain whitespace or "
                                 "control characters")
    return text


def normalise_display_name(value: Any) -> str:
    """One bounded line of display text (may be empty: "no display name")."""
    if not isinstance(value, str):
        raise UserStoreError("display name must be a string")
    text = " ".join(value.split())
    if len(text) > DISPLAY_NAME_MAX_CHARS:
        raise UserStoreError(
            f"display name must be at most {DISPLAY_NAME_MAX_CHARS} characters")
    return text


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class UserStore:
    """The allowlist plus its permissions, in one atomically written file.

    Every read re-reads the file: the owner may edit it by hand, and a page
    that shows a permission the file does not hold is a page that lies. The
    file is small and administrator actions are rare, so this costs nothing.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(os.path.realpath(os.fspath(path)))
        self._lock = threading.RLock()

    # -- file ------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise UserStoreError(
                f"the user store is unreadable: {exc.strerror or exc}") from None
        if len(raw) > MAX_USERS_BYTES:
            raise UserStoreError("the user store is larger than it can be")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise UserStoreError("the user store is not valid JSON") from None
        if not isinstance(document, dict) or document.get("schema") != USERS_SCHEMA:
            raise UserStoreError("the user store does not carry its schema")
        users = document.get("users")
        if not isinstance(users, dict) or set(document) != {"schema", "users"}:
            raise UserStoreError("the user store is malformed")
        out: dict[str, dict[str, Any]] = {}
        for email, entry in users.items():
            if not isinstance(entry, dict):
                raise UserStoreError("the user store is malformed")
            try:
                key = normalise_email(email)
            except UserStoreError:
                raise UserStoreError("the user store holds an invalid email") from None
            sub = entry.get("sub")
            if sub is not None and not isinstance(sub, str):
                raise UserStoreError("the user store holds an invalid account id")
            allow = entry.get("allow")
            if not isinstance(allow, dict) or set(allow) != set(PERMISSIONS):
                raise UserStoreError("the user store holds invalid permissions")
            if any(type(flag) is not bool for flag in allow.values()):
                raise UserStoreError("the user store holds invalid permissions")
            try:
                display = normalise_display_name(entry.get("display_name", ""))
            except UserStoreError:
                raise UserStoreError(
                    "the user store holds an invalid display name") from None
            out[key] = {"sub": sub,
                        "added": str(entry.get("added") or ""),
                        "display_name": display,
                        "allow": {name: allow[name] for name in PERMISSIONS}}
        return out

    def _save(self, users: dict[str, dict[str, Any]]) -> None:
        document = {"schema": USERS_SCHEMA,
                    "users": {email: users[email] for email in sorted(users)}}
        payload = json.dumps(document, sort_keys=True,
                             ensure_ascii=False).encode("utf-8")
        directory = self.path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
            descriptor, temp_path = tempfile.mkstemp(prefix=".users-",
                                                    suffix=".tmp",
                                                    dir=str(directory))
        except OSError as exc:
            raise UserStoreError(
                f"the user store cannot be written: {exc.strerror or exc}") from None
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        except BaseException as exc:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            if isinstance(exc, OSError):
                raise UserStoreError(
                    f"the user store cannot be written: {exc.strerror or exc}") \
                    from None
            raise

    # -- reads -----------------------------------------------------------

    def list_users(self) -> list[dict[str, Any]]:
        """Every invited user, in email order, as plain JSON-able dicts."""
        with self._lock:
            users = self._load()
        return [{"email": email,
                 "sub": users[email]["sub"],
                 "added": users[email]["added"],
                 "display_name": users[email]["display_name"],
                 "allow": dict(users[email]["allow"])}
                for email in sorted(users)]

    def get(self, email: str) -> dict[str, Any] | None:
        key = normalise_email(email)
        with self._lock:
            entry = self._load().get(key)
        if entry is None:
            return None
        return {"email": key, "sub": entry["sub"], "added": entry["added"],
                "display_name": entry["display_name"],
                "allow": dict(entry["allow"])}

    def is_invited(self, email: str) -> bool:
        try:
            key = normalise_email(email)
        except UserStoreError:
            return False
        with self._lock:
            return key in self._load()

    def permissions(self, email: str) -> dict[str, bool]:
        """The four flags, or the all-false set for an unknown email: a caller
        must never gain a permission by naming an account that is not there."""
        key = normalise_email(email)
        with self._lock:
            entry = self._load().get(key)
        if entry is None:
            return {name: False for name in PERMISSIONS}
        return dict(entry["allow"])

    # -- writes ----------------------------------------------------------

    def add(self, email: str) -> dict[str, Any]:
        """Invite one account (q8 only). Re-adding keeps the existing flags."""
        key = normalise_email(email)
        with self._lock:
            users = self._load()
            if key not in users:
                if len(users) >= MAX_USERS:
                    raise UserStoreError(
                        f"the user store holds at most {MAX_USERS} accounts")
                users[key] = {"sub": None, "added": _now(), "display_name": "",
                              "allow": dict(DEFAULT_ALLOW)}
                self._save(users)
            entry = users[key]
        return {"email": key, "sub": entry["sub"], "added": entry["added"],
                "display_name": entry["display_name"],
                "allow": dict(entry["allow"])}

    def set_permissions(self, email: str, **flags: bool) -> dict[str, Any]:
        """Change named flags only; an unknown flag is a typed refusal."""
        key = normalise_email(email)
        unknown = sorted(set(flags) - set(PERMISSIONS))
        if unknown:
            raise UserStoreError("unknown permission(s): " + ", ".join(unknown))
        for name, value in flags.items():
            if type(value) is not bool:
                raise UserStoreError(f"permission {name} must be boolean")
        with self._lock:
            users = self._load()
            entry = users.get(key)
            if entry is None:
                raise UserStoreError("that account is not invited")
            entry["allow"].update(flags)
            self._save(users)
        return {"email": key, "sub": entry["sub"], "added": entry["added"],
                "display_name": entry["display_name"],
                "allow": dict(entry["allow"])}

    def set_display_name(self, email: str, name: str) -> dict[str, Any]:
        key = normalise_email(email)
        text = normalise_display_name(name)
        with self._lock:
            users = self._load()
            entry = users.get(key)
            if entry is None:
                raise UserStoreError("that account is not invited")
            entry["display_name"] = text
            self._save(users)
        return {"email": key, "sub": entry["sub"], "added": entry["added"],
                "display_name": entry["display_name"],
                "allow": dict(entry["allow"])}

    def remove(self, email: str, delete_data: bool) -> dict[str, Any]:
        """Uninvite one account. ``delete_data`` records the caller's intent for
        the job and reference stores, which this module does not own."""
        if type(delete_data) is not bool:
            raise UserStoreError("delete_data must be boolean")
        key = normalise_email(email)
        with self._lock:
            users = self._load()
            entry = users.pop(key, None)
            if entry is None:
                raise UserStoreError("that account is not invited")
            self._save(users)
        return {"email": key, "delete_data": delete_data,
                "sub": entry["sub"]}

    def bind_sub(self, email: str, sub: str) -> dict[str, Any]:
        """Record the Google account id on first sign-in; later sign-ins must
        present the same one."""
        key = normalise_email(email)
        if not isinstance(sub, str) or not sub.strip():
            raise UserStoreError("account id must be a non-empty string")
        identity = sub.strip()
        with self._lock:
            users = self._load()
            entry = users.get(key)
            if entry is None:
                raise UserStoreError("that account is not invited")
            if entry["sub"] is None:
                entry["sub"] = identity
                self._save(users)
            elif entry["sub"] != identity:
                raise UserStoreError("that account is already bound to another "
                                     "sign-in")
        return {"email": key, "sub": entry["sub"], "added": entry["added"],
                "display_name": entry["display_name"],
                "allow": dict(entry["allow"])}
