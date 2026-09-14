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
import ipaddress
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
# High-priority clients (Tanzu talk / nest) preempt bulk (Paperclip, UIs).
# One Ollama slot. A bulk 32k prompt otherwise 400s the talk at ~60s.
DEMO_CIDRS_RAW = os.environ.get("LLAMAGATE_DEMO_CIDRS", "172.16.0.0/16")
DEMO_KEYS_RAW = os.environ.get("LLAMAGATE_DEMO_KEYS", "ollama")
DEMO_LEASE_SEC = int(os.environ.get("LLAMAGATE_DEMO_LEASE_SEC", "90"))
BULK_WAIT_SEC = int(os.environ.get("LLAMAGATE_BULK_WAIT_SEC", "180"))
# Talk turns queue behind each other. 30s was too short: kicking off a
# 29k Paperclip prompt can take longer than that, and Agent Builder
# then showed 503 "high-priority" which is not a Grant revoke.
DEMO_WAIT_SEC = int(os.environ.get("LLAMAGATE_DEMO_WAIT_SEC", "180"))

_lock = threading.Lock()
GPU_PATHS = {
    "v1/chat/completions",
    "v1/completions",
    "api/chat",
    "api/generate",
}


def _parse_cidrs(raw):
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return out


def _parse_keys(raw):
    return {p.strip() for p in (raw or "").split(",") if p.strip()}


DEMO_CIDRS = _parse_cidrs(DEMO_CIDRS_RAW)
DEMO_KEYS = _parse_keys(DEMO_KEYS_RAW)


def _bearer_token(authorization):
    raw = (authorization or "").strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def classify_client(remote_addr, authorization="", class_header=""):
    """high = talk/demo. bulk = Paperclip and everything else."""
    header = (class_header or "").strip().lower()
    if header in {"high", "demo"}:
        return "high"
    if header in {"bulk", "low", "paperclip"}:
        return "bulk"
    token = _bearer_token(authorization)
    if token and token in DEMO_KEYS:
        return "high"
    try:
        ip = ipaddress.ip_address((remote_addr or "").split("%")[0])
    except ValueError:
        return "bulk"
    for net in DEMO_CIDRS:
        if ip in net:
            return "high"
    return "bulk"


class Preempted(Exception):
    """Bulk call was cancelled so a high-priority client can run."""


class GpuSlot:
    """One generation slot. high preempts bulk. high lease keeps bulk out
    between talk turns so Agent Builder can call a tool and speak."""

    def __init__(self, lease_sec=DEMO_LEASE_SEC):
        self.cv = threading.Condition()
        self.holder = None
        self.ticket = 0
        self.cancelled = False
        self.session = None
        self.upstream = None
        self.demo_until = 0.0
        self.preempts = 0
        self.lease_sec = int(lease_sec)

    def snapshot(self):
        with self.cv:
            remaining = max(0.0, self.demo_until - time.time())
            return {
                "holder": self.holder,
                "demo_lease_remaining_sec": round(remaining, 1),
                "preempts": self.preempts,
            }

    def acquire(self, cls, wait_sec):
        deadline = time.time() + max(0.0, float(wait_sec))
        with self.cv:
            while True:
                now = time.time()
                if cls == "high":
                    if self.holder == "bulk":
                        if not self.cancelled:
                            self._cancel_locked()
                    elif self.holder is None:
                        return self._take_locked("high")
                elif self.holder is None and now >= self.demo_until:
                    return self._take_locked("bulk")
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.cv.wait(timeout=min(remaining, 0.25))

    def _take_locked(self, cls):
        self.ticket += 1
        self.holder = cls
        self.cancelled = False
        self.session = None
        self.upstream = None
        if cls == "high":
            self.demo_until = time.time() + self.lease_sec
        return self.ticket

    def _cancel_locked(self):
        self.cancelled = True
        self.preempts += 1
        if self.upstream is not None:
            try:
                self.upstream.close()
            except Exception:
                pass
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.cv.notify_all()

    def bind(self, ticket, session=None, upstream=None):
        with self.cv:
            if self.ticket != ticket:
                return
            if session is not None:
                self.session = session
            if upstream is not None:
                self.upstream = upstream

    def throw_if_cancelled(self, ticket):
        with self.cv:
            if self.ticket != ticket or self.cancelled:
                raise Preempted()

    def release(self, ticket):
        with self.cv:
            if self.ticket != ticket:
                return
            if self.holder == "high":
                self.demo_until = time.time() + self.lease_sec
            self.holder = None
            self.cancelled = False
            self.session = None
            self.upstream = None
            self.cv.notify_all()


SLOT = GpuSlot()


def _busy_message(cls, holder=None):
    """Ordinary words for the talk UI. Not a Grant. Not IAM."""
    if cls == "high":
        if holder == "bulk":
            return "The model is still finishing a background job. Ask again in a few seconds."
        if holder == "high":
            return "The model is still answering the last talk turn. Ask again in a few seconds."
        return "The model is finishing another request. Ask again in a few seconds."
    return "The model is reserved for a live talk. Retry shortly."


def _busy_response(cls):
    holder = SLOT.snapshot().get("holder")
    msg = _busy_message(cls, holder)
    status = 503 if cls == "high" else 429
    body = json.dumps(
        {"error": {"message": msg, "type": "unavailable", "code": "gpu_busy"}}
    )
    return Response(
        body,
        status=status,
        content_type="application/json",
        headers={"Retry-After": "15"},
    )


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
    """OpenAI-compatible SSE. Count usage once — the last usage object.

    Some servers (and our own include_usage injection) repeat a running
    usage on many frames. Adding every frame made the public counter
    read tens of millions in a day while the company ledger stayed in
    the hundreds of thousands.
    """
    try:
        text = chunk_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return 0
    last = 0
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
        if not isinstance(usage, dict):
            continue
        last = int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("completion_tokens", 0) or 0
        )
    return last


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
def _client_class():
    return classify_client(
        request.remote_addr,
        request.headers.get("Authorization", ""),
        request.headers.get("X-Llamagate-Class", ""),
    )


@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    url = f"{OLLAMA_UPSTREAM}/{path}"
    clean_path = path.rstrip("/")
    raw_body = request.get_data()
    if clean_path in SSE_COUNTABLE_PATHS and request.method == "POST":
        raw_body = _ensure_stream_usage(raw_body)

    gated = clean_path in GPU_PATHS and request.method == "POST"
    cls = _client_class() if gated else None
    ticket = None
    session = None
    if gated:
        wait = DEMO_WAIT_SEC if cls == "high" else BULK_WAIT_SEC
        ticket = SLOT.acquire(cls, wait)
        if ticket is None:
            return _busy_response(cls)
        session = requests.Session()
        SLOT.bind(ticket, session=session)

    try:
        if session is not None:
            SLOT.throw_if_cancelled(ticket)
            upstream_resp = session.request(
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
            SLOT.bind(ticket, upstream=upstream_resp)
            SLOT.throw_if_cancelled(ticket)
        else:
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
    except Preempted:
        if ticket is not None:
            SLOT.release(ticket)
        if session is not None:
            session.close()
        body = json.dumps(
            {
                "error": {
                    "message": "Preempted by a high-priority client",
                    "type": "unavailable",
                    "code": "preempted",
                }
            }
        )
        return Response(body, status=499, content_type="application/json")
    except Exception:
        if ticket is not None:
            SLOT.release(ticket)
        if session is not None:
            session.close()
        raise

    def _finish():
        if ticket is not None:
            SLOT.release(ticket)
        if session is not None:
            session.close()

    if clean_path in NDJSON_COUNTABLE_PATHS:
        def generate():
            total = 0
            try:
                for line in upstream_resp.iter_lines():
                    if ticket is not None:
                        SLOT.throw_if_cancelled(ticket)
                    if line:
                        total += _extract_tokens_from_ndjson_line(line)
                        yield line + b"\n"
                if total > 0:
                    _add_tokens(total)
            except Preempted:
                return
            finally:
                _finish()

    elif clean_path in SSE_COUNTABLE_PATHS:
        def generate():
            # Buffer the body. Count once from the finished stream so a
            # usage object split across 1 KiB chunks, or repeated on
            # every frame, is not added over and over.
            buf = bytearray()
            try:
                for chunk in upstream_resp.iter_content(chunk_size=1024):
                    if ticket is not None:
                        SLOT.throw_if_cancelled(ticket)
                    if chunk:
                        buf.extend(chunk)
                        yield chunk
                body = bytes(buf)
                total = _extract_tokens_from_sse_chunk(body)
                if total <= 0:
                    total = _extract_tokens_from_openai_json(body)
                if total > 0:
                    _add_tokens(total)
            except Preempted:
                return
            finally:
                _finish()

    else:
        # Raw passthrough. Deliberately does NOT use iter_lines() here -
        # iter_lines() silently drops blank lines, which corrupts SSE
        # framing (blank lines are meaningful event separators in SSE,
        # not noise). Any path we don't explicitly count still needs to
        # be forwarded byte-for-byte intact.
        def generate():
            try:
                for chunk in upstream_resp.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            finally:
                _finish()

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


@app.route("/proxy/slot")
def slot_status():
    """Who holds the generation slot. No client identities or keys."""
    snap = SLOT.snapshot()
    snap["ok"] = True
    return snap


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
