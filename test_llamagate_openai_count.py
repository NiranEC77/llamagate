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


if __name__ == "__main__":
    test_nonstream_json()
    test_sse_with_usage()
    test_sse_repeated_usage_counts_once()
    test_sse_without_usage()
    test_inject_stream_usage()
    test_inject_noop_when_already_set()
    test_inject_noop_when_not_streaming()
    print("OK")
