#!/usr/bin/env python3
"""Unit tests for high-priority vs bulk GPU slot (no live Ollama)."""
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

flask = types.ModuleType("flask")


class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def deco(fn):
            return fn

        return deco


flask.Flask = _Flask
flask.request = None
flask.Response = object
flask.stream_with_context = lambda x: x
sys.modules["flask"] = flask
sys.modules.setdefault("requests", types.ModuleType("requests"))

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("llamagate", ROOT / "llamagate.py")
lg = importlib.util.module_from_spec(spec)
sys.modules["llamagate"] = lg
spec.loader.exec_module(lg)


def test_classify_nest_and_key_are_high():
    assert lg.classify_client("172.16.19.60") == "high"
    assert lg.classify_client("192.168.1.194", "Bearer ollama") == "high"
    assert lg.classify_client("192.168.1.194", "Bearer OLLAMA") == "bulk"
    assert lg.classify_client("192.168.1.194") == "bulk"
    assert lg.classify_client("127.0.0.1") == "bulk"
    assert lg.classify_client("10.0.0.2", "", "demo") == "high"
    assert lg.classify_client("172.16.19.60", "", "bulk") == "bulk"


def test_bulk_waits_for_high_then_runs():
    slot = lg.GpuSlot(lease_sec=0)
    t_high = slot.acquire("high", 1)
    assert t_high is not None
    got = []

    def waiter():
        got.append(slot.acquire("bulk", 2))

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)
    assert got == []
    slot.release(t_high)
    th.join(timeout=2)
    assert got and got[0] is not None
    slot.release(got[0])


def test_high_preempts_bulk():
    slot = lg.GpuSlot(lease_sec=0)
    t_bulk = slot.acquire("bulk", 1)
    assert t_bulk is not None
    high_ticket = []

    def taker():
        high_ticket.append(slot.acquire("high", 2))

    th = threading.Thread(target=taker)
    th.start()
    time.sleep(0.05)
    try:
        slot.throw_if_cancelled(t_bulk)
        raise AssertionError("bulk should have been cancelled")
    except lg.Preempted:
        slot.release(t_bulk)
    th.join(timeout=2)
    assert high_ticket and high_ticket[0] is not None
    snap = slot.snapshot()
    assert snap["preempts"] == 1
    slot.release(high_ticket[0])


def test_busy_message_is_not_grant_jargon():
    for holder in ("bulk", "high", None):
        msg = lg._busy_message("high", holder)
        assert "high-priority" not in msg.lower()
        assert "grant" not in msg.lower()
        assert "denied" not in msg.lower()
    assert "background job" in lg._busy_message("high", "bulk")


def test_high_lease_blocks_bulk():
    slot = lg.GpuSlot(lease_sec=1)
    t_high = slot.acquire("high", 1)
    slot.release(t_high)
    assert slot.acquire("bulk", 0.15) is None
    time.sleep(1.1)
    t_bulk = slot.acquire("bulk", 1)
    assert t_bulk is not None
    slot.release(t_bulk)


def test_demo_lock_refuses_bulk_and_keeps_high():
    slot = lg.GpuSlot(lease_sec=0)
    assert slot.set_demo_lock(True) is True
    assert slot.snapshot()["demo_lock"] is True
    assert slot.acquire("bulk", 0.2) is None
    t_high = slot.acquire("high", 1)
    assert t_high is not None
    slot.release(t_high)
    slot.set_demo_lock(False)
    t_bulk = slot.acquire("bulk", 1)
    assert t_bulk is not None
    slot.release(t_bulk)


def test_demo_lock_cancels_running_bulk():
    slot = lg.GpuSlot(lease_sec=0)
    t_bulk = slot.acquire("bulk", 1)
    assert t_bulk is not None
    slot.set_demo_lock(True)
    try:
        slot.throw_if_cancelled(t_bulk)
        raise AssertionError("bulk should have been cancelled")
    except lg.Preempted:
        slot.release(t_bulk)
    assert slot.snapshot()["demo_lock"] is True
    slot.set_demo_lock(False)


def test_busy_message_names_demo_lock():
    msg = lg._busy_message("bulk", demo_lock=True)
    assert "demo" in msg.lower()
    assert "grant" not in msg.lower()


def test_waiting_counts_show_queue():
    slot = lg.GpuSlot(lease_sec=0)
    t_high = slot.acquire("high", 1, actor="talk")
    assert t_high is not None
    snap = slot.snapshot()
    assert snap["busy"] is True
    assert snap["holder_name"] == "Tanzu talk"
    got = []

    def waiter():
        got.append(slot.acquire("bulk", 2, actor="it-hermes"))

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.1)
    queued = slot.snapshot()
    assert queued["waiting_other"] == 1
    assert queued["waiting"] == 1
    slot.release(t_high)
    th.join(timeout=2)
    assert got and got[0] is not None
    after = slot.snapshot()
    assert after["holder_name"] == "IT"
    slot.release(got[0])
    assert slot.snapshot()["waiting"] == 0
    assert slot.snapshot()["busy"] is False


def test_holder_records_process_start_and_agent():
    slot = lg.GpuSlot(lease_sec=0)
    assert slot.snapshot()["holder_since"] is None
    assert slot.snapshot()["next"] is None
    t = slot.acquire("bulk", 1, actor="it-hermes", process="chat on gpt-oss:120b")
    snap = slot.snapshot()
    assert snap["holder_name"] == "IT"
    assert snap["holder_process"] == "chat on gpt-oss:120b"
    assert snap["holder_since"].endswith("Z")
    assert snap["holder_seconds"] >= 0
    slot.release(t)
    after = slot.snapshot()
    assert after["holder_process"] is None
    assert after["holder_since"] is None


def test_queue_names_next_in_order_and_serves_it():
    slot = lg.GpuSlot(lease_sec=0)
    t_hold = slot.acquire("bulk", 1, actor="brain", process="chat on m")
    got = {}

    def waiter(actor, cls):
        got[actor] = slot.acquire(cls, 3, actor=actor, process="chat on m")
        if got[actor] is not None:
            got.setdefault("order", []).append(actor)
            time.sleep(0.05)
            slot.release(got[actor])

    th_sec = threading.Thread(target=waiter, args=("security-hermes", "bulk"))
    th_sec.start()
    time.sleep(0.05)
    th_mkt = threading.Thread(target=waiter, args=("marketing-hermes", "bulk"))
    th_mkt.start()
    time.sleep(0.05)
    snap = slot.snapshot()
    assert [w["name"] for w in snap["queue"]] == ["Security", "Marketing"]
    assert snap["next"]["name"] == "Security"
    assert snap["next"]["process"] == "chat on m"
    assert snap["next"]["since"].endswith("Z")
    slot.release(t_hold)
    th_sec.join(timeout=3)
    th_mkt.join(timeout=3)
    assert got["order"] == ["security-hermes", "marketing-hermes"]
    assert slot.snapshot()["queue"] == []


def test_talk_waiter_is_next_ahead_of_earlier_bulk():
    slot = lg.GpuSlot(lease_sec=0)
    t_hold = slot.acquire("high", 1, actor="talk")
    th_bulk = threading.Thread(target=lambda: slot.acquire("bulk", 0.5, actor="it-hermes"))
    th_bulk.start()
    time.sleep(0.05)
    th_high = threading.Thread(target=lambda: slot.acquire("high", 0.5, actor="talk"))
    th_high.start()
    time.sleep(0.05)
    snap = slot.snapshot()
    assert snap["next"]["name"] == "Tanzu talk"
    assert [w["class"] for w in snap["queue"]] == ["high", "bulk"]
    th_bulk.join(timeout=2)
    th_high.join(timeout=2)
    slot.release(t_hold)


def test_describe_process_reads_model_not_prompt():
    body = b'{"model": "gpt-oss:120b", "messages": [{"role": "user", "content": "secret"}]}'
    assert lg.describe_process("v1/chat/completions", body) == "chat on gpt-oss:120b"
    assert lg.describe_process("api/generate", b'{"model":"qwen3:32b"}') == "generate on qwen3:32b"
    assert lg.describe_process("api/chat", b"not json") == "chat"
    assert "secret" not in lg.describe_process("api/chat", body)


def test_demo_lock_persists():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "demo_lock.json")
        slot = lg.GpuSlot(lease_sec=0, lock_path=path)
        assert slot.set_demo_lock(True) is True
        again = lg.GpuSlot(lease_sec=0, lock_path=path)
        assert again.snapshot()["demo_lock"] is True


if __name__ == "__main__":
    test_classify_nest_and_key_are_high()
    test_bulk_waits_for_high_then_runs()
    test_high_preempts_bulk()
    test_busy_message_is_not_grant_jargon()
    test_high_lease_blocks_bulk()
    test_demo_lock_refuses_bulk_and_keeps_high()
    test_demo_lock_cancels_running_bulk()
    test_busy_message_names_demo_lock()
    test_waiting_counts_show_queue()
    test_holder_records_process_start_and_agent()
    test_queue_names_next_in_order_and_serves_it()
    test_talk_waiter_is_next_ahead_of_earlier_bulk()
    test_describe_process_reads_model_not_prompt()
    test_demo_lock_persists()
    print("OK")
