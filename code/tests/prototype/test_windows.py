import json
from pathlib import Path

from ictc.prototype.windows import run_loopback_prototype


def test_loopback_prototype_records_route_change_fallback_and_real_workers(tmp_path: Path) -> None:
    events = run_loopback_prototype(tmp_path / "m12")

    assert len(events) == 16
    assert [event.event for event in events if event.component == "foreground"] == [
        "PACKET_DELIVERED"
    ] * 6
    assert any(event.event == "ROUTE_CHANGE" for event in events)
    assert any(event.event == "TELEMETRY_BREACH" for event in events)
    assert any(event.event == "BREACH_FALLBACK" for event in events)
    assert [
        event.detail
        for event in events
        if event.component == "background" and event.event == "PACKET_DELIVERED"
    ] == [
        "background:0",
        "background:1",
        "background:2",
    ]

    rows = [
        json.loads(line)
        for line in (tmp_path / "m12" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(events)
