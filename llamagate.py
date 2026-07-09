"""
llamagate
=========

A transparent gateway that sits in front of Ollama.

Forwards every request byte-for-byte unchanged, so it's a drop-in
replacement for Ollama's own address - point any existing client at
llamagate instead of Ollama directly, and it keeps working exactly as
before. Llamagate observes traffic in transit to add capabilities Ollama
doesn't have natively.

Today: token usage counting (today + all-time), persisted to a JSON file,
safe across concurrent requests.

Roadmap: llamagate is designed as the foundation for an observability and
security layer in front of Ollama - request logging, per-client usage
breakdowns, rate limiting, and access control are natural next additions
without changing how clients connect. See "Adding a new feature" in
README.md.
"""

from flask import Flask, request, Response, stream_with_context
import requests
import json
import os
import threading
from datetime import date

app = Flask(__name__)

# --------------------------------------------------------------------------
# Configuration - all overridable via environment variables.
# --------------------------------------------------------------------------
OLLAMA_UPSTREAM = os.environ.get("OLLAMA_UPSTREAM_URL", "http://127.0.0.1:11434")
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11434"))
COUNTS_FILE = os.environ.get(
    "LLAMAGATE_COUNTS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_counts.json"),
)
UPSTREAM_TIMEOUT = int(os.environ.get("LLAMAGATE_UPSTREAM_TIMEOUT", "300"))

_lock = threading.Lock()


# --------------------------------------------------------------------------
# Token counting - persisted state
# --------------------------------------------------------------------------
def _load_counts():
    if not os.path.exists(COUNTS_FILE):
        return {"date": str(date.today()), "tokens_today": 0, "tokens_total": 0}
    try:
        with open(COUNTS_FILE) as f:
            data = json.load(f)
    except Exception:
        return {"date": str(date.today()), "tokens_today": 0, "tokens_total": 0}

    if data.get("date") != str(date.today()):
        data["date"] = str(date.today())
        data["tokens_today"] = 0
    return data


def _save_counts(data):
    with open(COUNTS_FILE, "w") as f:
        json.dump(data, f)


def _add_tokens(n):
    if n <= 0:
        return
    with _lock:
        data = _load_counts()
        data["tokens_today"] += n
        data["tokens_total"] += n
        _save_counts(data)


def get_token_counts():
    """Public helper - importable by other services (e.g. a stats/dashboard
    process) that want to read counts without going through HTTP."""
    data = _load_counts()
    return {"tokens_today": data["tokens_today"], "tokens_total": data["tokens_total"]}


# --------------------------------------------------------------------------
# Response parsers for the two streaming formats Ollama-compatible clients use
# --------------------------------------------------------------------------
def _extract_tokens_from_ndjson_line(line):
    """Ollama's native API (/api/generate, /api/chat) streams newline-
    delimited JSON. The final chunk (done: true) carries token counts."""
    try:
        obj = json.loads(line)
    except Exception:
        return 0
    if not obj.get("done"):
        return 0
    return obj.get("eval_count", 0) + obj.get("prompt_eval_count", 0)


def _extract_tokens_from_sse_chunk(chunk_bytes):
    """OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions)
    stream Server-Sent Events. A 'usage' field, when present, is usually
    only on the final chunk (and only if the client requested it)."""
    try:
        text = chunk_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return 0
    total = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]" or not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        usage = obj.get("usage")
        if usage:
            total += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return total


# --------------------------------------------------------------------------
# Path classification - which paths get token-counted, and how
# --------------------------------------------------------------------------
NDJSON_COUNTABLE_PATHS = {"api/generate", "api/chat"}
SSE_COUNTABLE_PATHS = {"v1/chat/completions", "v1/completions"}


# --------------------------------------------------------------------------
# Core proxy route
# --------------------------------------------------------------------------
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    url = f"{OLLAMA_UPSTREAM}/{path}"
    upstream_resp = requests.request(
        method=request.method,
        url=url,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        data=request.get_data(),
        params=request.args,
        stream=True,
        timeout=UPSTREAM_TIMEOUT,
    )

    clean_path = path.rstrip("/")

    if clean_path in NDJSON_COUNTABLE_PATHS:
        def generate():
            total = 0
            for line in upstream_resp.iter_lines():
                if line:
                    total += _extract_tokens_from_ndjson_line(line)
                    yield line + b"\n"
            if total > 0:
                _add_tokens(total)

    elif clean_path in SSE_COUNTABLE_PATHS:
        def generate():
            total = 0
            for chunk in upstream_resp.iter_content(chunk_size=1024):
                if chunk:
                    total += _extract_tokens_from_sse_chunk(chunk)
                    yield chunk
            if total > 0:
                _add_tokens(total)

    else:
        # Raw passthrough. Deliberately does NOT use iter_lines() here -
        # iter_lines() silently drops blank lines, which corrupts SSE
        # framing (blank lines are meaningful event separators in SSE,
        # not noise). Any path we don't explicitly count still needs to
        # be forwarded byte-for-byte intact.
        def generate():
            for chunk in upstream_resp.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

    return Response(
        stream_with_context(generate()),
        status=upstream_resp.status_code,
        content_type=upstream_resp.headers.get("Content-Type", "application/json"),
    )


# --------------------------------------------------------------------------
# Proxy's own endpoints (not forwarded upstream)
# --------------------------------------------------------------------------
@app.route("/proxy/health")
def health():
    return {"ok": True, "upstream": OLLAMA_UPSTREAM}


@app.route("/proxy/stats")
def stats():
    return get_token_counts()


if __name__ == "__main__":
    app.run(host=PROXY_HOST, port=PROXY_PORT)
