import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asana_client as ac  # noqa: E402


class FakeResp:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.ok, self.text = body, status, status < 400, str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.headers, self.responses, self.calls = {}, list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.responses.pop(0)


def client(responses, ws="1"):
    return ac.AsanaClient("tok", ws, session=FakeSession(responses))


def test_my_tasks_paginates_and_sorts():
    c = client([
        FakeResp({"data": [{"gid": "1", "name": "b", "due_on": None}], "next_page": {"offset": "x"}}),
        FakeResp({"data": [{"gid": "2", "name": "a", "due_on": "2026-01-02"},
                           {"gid": "3", "name": "done", "completed": True}], "next_page": None}),
    ])
    tasks = c.my_tasks()
    assert [t["gid"] for t in tasks] == ["2", "1"]
    assert c.session.calls[1][2]["params"]["offset"] == "x"


def test_workspace_autodetect():
    c = client([FakeResp({"data": {"workspaces": [{"gid": "99"}]}})], ws=None)
    assert c.workspace() == "99"


def test_create_and_complete():
    c = client([FakeResp({"data": {"gid": "5", "name": "n"}}), FakeResp({"data": {"gid": "5", "name": "n"}})])
    c.create_task("n", "2026-12-31")
    c.complete_task("5")
    assert c.session.calls[0][2]["json"]["data"] == {"name": "n", "assignee": "me", "workspace": "1", "due_on": "2026-12-31"}
    assert c.session.calls[1][2]["json"] == {"data": {"completed": True}}


def test_error_surface():
    c = client([FakeResp({"errors": [{"message": "Not Authorized"}]}, 401)])
    with pytest.raises(ac.AsanaError, match="401: Not Authorized"):
        c.my_tasks()


def test_filters():
    ts = [{"name": "Pay Rent", "due_on": "2026-10-01"}, {"name": "Call", "due_on": "2026-10-05"}, {"name": "x"}]
    d = dt.date(2026, 10, 5)
    assert [t["name"] for t in ac.overdue(ts, d)] == ["Pay Rent"]
    assert [t["name"] for t in ac.due_today(ts, d)] == ["Call"]
    assert [t["name"] for t in ac.find(ts, "rent")] == ["Pay Rent"]
