"""Minimal Asana REST client (Personal Access Token auth)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Iterator

import requests

BASE_URL = "https://app.asana.com/api/1.0"
TASK_FIELDS = "name,due_on,completed,permalink_url,projects.name"


class AsanaError(RuntimeError):
    pass


class AsanaClient:
    def __init__(self, token: str, workspace_gid: str | None = None, session: requests.Session | None = None):
        self.workspace_gid = workspace_gid or None
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        resp = self.session.request(method, f"{BASE_URL}{path}", timeout=20, **kwargs)
        if not resp.ok:
            try:
                msg = "; ".join(e.get("message", "") for e in resp.json().get("errors", []))
            except ValueError:
                msg = resp.text[:200]
            raise AsanaError(f"Asana {resp.status_code}: {msg}")
        return resp.json()

    def _paginate(self, path: str, params: dict) -> Iterator[dict]:
        params = {**params, "limit": 100}
        while True:
            body = self._request("GET", path, params=params)
            yield from body["data"]
            nxt = body.get("next_page")
            if not nxt:
                return
            params["offset"] = nxt["offset"]

    def workspace(self) -> str:
        if not self.workspace_gid:
            spaces = self._request("GET", "/users/me")["data"]["workspaces"]
            if not spaces:
                raise AsanaError("No Asana workspace found for this token")
            self.workspace_gid = spaces[0]["gid"]
        return self.workspace_gid

    def my_tasks(self) -> list[dict]:
        """Incomplete tasks assigned to me, soonest due first (undated last)."""
        tasks = list(self._paginate("/tasks", {
            "assignee": "me", "workspace": self.workspace(),
            "completed_since": "now", "opt_fields": TASK_FIELDS,
        }))
        tasks = [t for t in tasks if not t.get("completed")]
        return sorted(tasks, key=lambda t: (t.get("due_on") is None, t.get("due_on") or ""))

    def create_task(self, name: str, due_on: str | None = None, notes: str | None = None) -> dict:
        data: dict[str, Any] = {"name": name, "assignee": "me", "workspace": self.workspace()}
        if due_on:
            data["due_on"] = due_on
        if notes:
            data["notes"] = notes
        return self._request("POST", "/tasks", json={"data": data}, params={"opt_fields": TASK_FIELDS})["data"]

    def complete_task(self, gid: str) -> dict:
        return self._request("PUT", f"/tasks/{gid}", json={"data": {"completed": True}},
                             params={"opt_fields": TASK_FIELDS})["data"]


def due_today(tasks: list[dict], today: dt.date | None = None) -> list[dict]:
    today = (today or dt.date.today()).isoformat()
    return [t for t in tasks if t.get("due_on") == today]


def overdue(tasks: list[dict], today: dt.date | None = None) -> list[dict]:
    today = (today or dt.date.today()).isoformat()
    return [t for t in tasks if t.get("due_on") and t["due_on"] < today]


def find(tasks: list[dict], text: str) -> list[dict]:
    text = text.lower()
    return [t for t in tasks if text in t["name"].lower()]
