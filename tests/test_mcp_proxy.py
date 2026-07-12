from __future__ import annotations

import pytest

from enfold.mcp_proxy import MemoryMCPProxy


class RecordingTransport:
    def __init__(self):
        self.calls = []

    def request(self, method, params=None, *, request_id=None):
        self.calls.append((method, params, request_id))
        return {"method": method, "params": params}


@pytest.mark.parametrize(
    ("proxy_method", "wire_method"),
    [
        ("write", "memory.write"),
        ("search", "memory.search"),
        ("evidence", "memory.evidence"),
        ("history", "memory.history"),
        ("changes", "memory.changes"),
        ("timeline", "memory.timeline"),
        ("entities", "memory.entities"),
        ("entity", "memory.entity"),
        ("conflicts", "memory.conflicts"),
        ("resolve_conflict", "memory.resolve_conflict"),
        ("enqueue_extraction", "memory.extraction.enqueue"),
    ],
)
def test_proxy_is_one_to_one_transport_adapter(proxy_method, wire_method):
    transport = RecordingTransport()
    proxy = MemoryMCPProxy(transport)

    result = getattr(proxy, proxy_method)({"value": 7})

    assert result == {"method": wire_method, "params": {"value": 7}}
    assert transport.calls == [(wire_method, {"value": 7}, None)]


def test_proxy_copies_params_and_rejects_non_mapping():
    transport = RecordingTransport()
    proxy = MemoryMCPProxy(transport)
    params = {"query": "current project"}

    proxy.search(params)
    params["query"] = "changed later"

    assert transport.calls[0][1] == {"query": "current project"}
    with pytest.raises(TypeError, match="mapping"):
        proxy.write("not an object")
