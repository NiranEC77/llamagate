#!/usr/bin/env python3
"""Unit tests for llamagate OpenAI token extraction (no Flask/server needed)."""
import importlib.util
import json
import sys
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


def test_nonstream_json():
    body = json.dumps(
        {
            "id": "x",
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    ).encode()
    assert lg._extract_tokens_from_openai_json(body) == 13


def test_sse_with_usage():
    chunk = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert lg._extract_tokens_from_sse_chunk(chunk) == 7


def test_sse_repeated_usage_counts_once():
    """Running totals on every frame must not be summed."""
    chunk = (
        b'data: {"choices":[{"delta":{"content":"a"}}],'
        b'"usage":{"prompt_tokens":6000,"completion_tokens":1}}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}],'
        b'"usage":{"prompt_tokens":6000,"completion_tokens":2}}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":6000,"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert lg._extract_tokens_from_sse_chunk(chunk) == 6003


def test_sse_without_usage():
    chunk = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    assert lg._extract_tokens_from_sse_chunk(chunk) == 0


def test_inject_stream_usage():
    raw = json.dumps(
        {"model": "gpt-oss:120b", "messages": [], "stream": True}
    ).encode()
    out = json.loads(lg._ensure_stream_usage(raw))
    assert out["stream"] is True
    assert out["stream_options"]["include_usage"] is True


def test_inject_noop_when_already_set():
    raw = json.dumps(
        {
            "model": "m",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    assert lg._ensure_stream_usage(raw) == raw


def test_inject_noop_when_not_streaming():
    raw = json.dumps({"model": "m", "stream": False}).encode()
    assert lg._ensure_stream_usage(raw) == raw


def test_sse_usage_split():
    chunk = (
        b'data: {"usage":{"prompt_tokens":29000,"completion_tokens":40}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert lg._extract_usage_from_sse_chunk(chunk) == (29000, 40)


def test_ndjson_last_done_wins():
    first = json.dumps(
        {"done": True, "prompt_eval_count": 10, "eval_count": 1}
    ).encode()
    last = json.dumps(
        {"done": True, "prompt_eval_count": 100, "eval_count": 5}
    ).encode()
    assert lg._extract_usage_from_ndjson_line(first) == (10, 1)
    assert lg._extract_usage_from_ndjson_line(last) == (100, 5)
    mid = json.dumps({"done": False, "eval_count": 2}).encode()
    assert lg._extract_usage_from_ndjson_line(mid) == (0, 0)


def test_embed_usage():
    body = json.dumps({"prompt_eval_count": 12, "embeddings": [[0.1]]}).encode()
    assert lg._extract_usage_from_embed_json(body) == (12, 0)
    openai = json.dumps({"usage": {"prompt_tokens": 8, "total_tokens": 8}}).encode()
    assert lg._extract_usage_from_embed_json(openai) == (8, 0)


def test_add_usage_splits_and_counts_once():
    import tempfile

    prev = lg.COUNTS_FILE
    try:
        with tempfile.TemporaryDirectory() as tmp:
            lg.COUNTS_FILE = str(Path(tmp) / "counts.json")
            lg._add_usage((100, 5), "bulk")
            lg._add_usage((20, 3), "high")
            counts = lg.get_token_counts()
            assert counts["tokens_today"] == 128
            assert counts["prompt_tokens_today"] == 120
            assert counts["completion_tokens_today"] == 8
            assert counts["bulk_tokens_today"] == 105
            assert counts["high_tokens_today"] == 23
            assert counts["requests_today"] == 2
            assert counts["tokens_total"] == 128
    finally:
        lg.COUNTS_FILE = prev


def test_day_roll_keeps_all_time():
    import tempfile

    prev = lg.COUNTS_FILE
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counts.json"
            path.write_text(
                json.dumps(
                    {
                        "date": "1999-01-01",
                        "tokens_today": 999,
                        "tokens_total": 5000,
                        "prompt_tokens_today": 800,
                        "completion_tokens_today": 199,
                    }
                )
            )
            lg.COUNTS_FILE = str(path)
            loaded = lg._load_counts()
            assert loaded["tokens_today"] == 0
            assert loaded["prompt_tokens_today"] == 0
            assert loaded["completion_tokens_today"] == 0
            assert loaded["tokens_total"] == 5000
    finally:
        lg.COUNTS_FILE = prev


if __name__ == "__main__":
    test_nonstream_json()
    test_sse_with_usage()
    test_sse_repeated_usage_counts_once()
    test_sse_without_usage()
    test_inject_stream_usage()
    test_inject_noop_when_already_set()
    test_inject_noop_when_not_streaming()
    test_sse_usage_split()
    test_ndjson_last_done_wins()
    test_embed_usage()
    test_add_usage_splits_and_counts_once()
    test_day_roll_keeps_all_time()
    print("OK")
