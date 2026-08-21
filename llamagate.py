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
safe across concurrent requests; GPU/model/rate telemetry for external
dashboards.

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
import subprocess
import threading
import time
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


def _tokens_per_sec(tokens_total):
    """Diff tokens_total against the last time /api/stats was polled. The
    sample lives in the same counts file (under the same lock) so the rate
    stays correct even if llamagate restarts or runs multiple workers -
    it's derived from durable state, not in-memory-only counters."""
    now = time.time()
    with _lock:
        data = _load_counts()
        prev_total = data.get("rate_sample_total")
        prev_ts = data.get("rate_sample_ts")
        data["rate_sample_total"] = tokens_total
        data["rate_sample_ts"] = now
        _save_counts(data)
    if prev_total is None or prev_ts is None:
        return None
    elapsed = now - prev_ts
    if elapsed <= 0:
        return None
    delta = tokens_total - prev_total
    if delta < 0:
        return None
    return round(delta / elapsed, 1)


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


def _extract_tokens_from_openai_json(body_bytes):
    """Non-streaming OpenAI JSON body: top-level ``usage`` object.

    This is what most clients (Tanzu GenAI tile, curl without stream:true,
    many SDKs) receive. The SSE parser never sees ``data:`` lines here, so
    without this path the public counter freezes while GPU telemetry stays
    live — exactly the 2026-08-17 niran.ai widget failure.
    """
    try:
        obj = json.loads(body_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return 0
    if not isinstance(obj, dict):
        return 0
    usage = obj.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("prompt_tokens", 0) or 0) + int(
        usage.get("completion_tokens", 0) or 0
    )


def _ensure_stream_usage(raw_body: bytes) -> bytes:
    """If the client asked for SSE streaming but omitted include_usage,
    inject it. Ollama (like OpenAI) only emits a usage frame when asked;
    without it the SSE counter always reads zero for stream:true traffic.

    Non-JSON / non-stream bodies are returned unchanged.
    """
    if not raw_body:
        return raw_body
    try:
        obj = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return raw_body
    if not isinstance(obj, dict) or not obj.get("stream"):
        return raw_body
    opts = obj.get("stream_options")
    if not isinstance(opts, dict):
        opts = {}
    if opts.get("include_usage") is True:
        return raw_body
    opts = dict(opts)
    opts["include_usage"] = True
    obj = dict(obj)
    obj["stream_options"] = opts
    return json.dumps(obj).encode("utf-8")


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
    clean_path = path.rstrip("/")
    raw_body = request.get_data()
    if clean_path in SSE_COUNTABLE_PATHS and request.method == "POST":
        raw_body = _ensure_stream_usage(raw_body)

    upstream_resp = requests.request(
        method=request.method,
        url=url,
        headers={
            k: v
            for k, v in request.headers
            if k.lower() not in ("host", "content-length")
        },
        data=raw_body,
        params=request.args,
        stream=True,
        timeout=UPSTREAM_TIMEOUT,
    )

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
            # Buffer the body so we can fall back to non-streaming JSON
            # parsing when the client did not use SSE (the common case).
            total = 0
            buf = bytearray()
            for chunk in upstream_resp.iter_content(chunk_size=1024):
                if chunk:
                    buf.extend(chunk)
                    total += _extract_tokens_from_sse_chunk(chunk)
                    yield chunk
            if total <= 0 and buf:
                total = _extract_tokens_from_openai_json(bytes(buf))
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


def _gpu_telemetry():
    """Query GPU utilization/temp/power via nvidia-smi. Returns None for
    every field if nvidia-smi isn't available or the query fails - the
    dashboard widget already renders '-' for null fields."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError(out.stderr.strip())
        util, temp, power = (p.strip() for p in out.stdout.strip().splitlines()[0].split(","))
        return {
            "gpu_util_pct": float(util),
            "gpu_temp_c": float(temp),
            "power_draw_w": float(power),
        }
    except Exception:
        return {"gpu_util_pct": None, "gpu_temp_c": None, "power_draw_w": None}


def _loaded_models():
    """Ask Ollama which models are currently resident in GPU memory (not
    just installed) via its native /api/ps. Falls back to an empty list -
    the widget already handles that gracefully."""
    try:
        resp = requests.get(f"{OLLAMA_UPSTREAM}/api/ps", timeout=3)
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            size = m.get("size")
            size_label = f"{size / 1e9:.1f}GB" if isinstance(size, (int, float)) else None
            name = m.get("name") or m.get("model")
            if name:
                models.append({"name": name, "size": size_label})
        return models
    except Exception:
        return []


@app.route("/api/stats")
def api_stats():
    """Aggregated status for external dashboards (the niran.ai DGX widget).
    Distinct from /proxy/stats (token counts only, used by other internal
    tooling) - this is the richer, public-facing shape."""
    counts = get_token_counts()
    return {
        "status": "online",
        "gpu": _gpu_telemetry(),
        "loaded_models": _loaded_models(),
        "tokens_today": counts["tokens_today"],
        "tokens_total": counts["tokens_total"],
        "tokens_per_sec": _tokens_per_sec(counts["tokens_total"]),
        "updated_at": int(time.time()),
    }


if __name__ == "__main__":
    app.run(host=PROXY_HOST, port=PROXY_PORT)
