#!/usr/bin/env bash
# Deploy the OpenAI-token-count fix for llamagate on spark1.
#
# Run ON spark1 as claude-orch (or as niranec with write access to the
# llamagate install). Safe: copies a file, syntax-checks, restarts the
# unit. Counts file is untouched.
#
# Usage:
#   ./deploy-llamagate-openai-count.sh
#   ./deploy-llamagate-openai-count.sh /path/to/live/llamagate.py

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/llamagate.py"
LIVE="${1:-}"

if [[ -z "$LIVE" ]]; then
  for cand in \
    /home/claude-orch/llamagate.py \
    /home/claude-orch/agent-lab/.planning/dgx-model-management/llamagate.py \
    /home/niranec/llamagate/llamagate.py \
    /opt/llamagate/llamagate.py
  do
    if [[ -f "$cand" ]]; then
      LIVE="$cand"
      break
    fi
  done
fi

if [[ -z "$LIVE" || ! -f "$LIVE" ]]; then
  echo "Could not find live llamagate.py. Pass the path:" >&2
  echo "  $0 /path/to/llamagate.py" >&2
  echo "Also: systemctl cat llamagate | grep ExecStart" >&2
  exit 1
fi

echo "Live file: $LIVE"
python3 -m py_compile "$SRC"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
cp -a "$LIVE" "${LIVE}.bak-${ts}"
cp "$SRC" "$LIVE"
python3 -m py_compile "$LIVE"

unit=""
for u in llamagate dgx-llamagate; do
  if systemctl cat "$u" >/dev/null 2>&1; then
    unit="$u"
    break
  fi
done

if [[ -n "$unit" ]]; then
  echo "Restarting $unit"
  if sudo -n systemctl restart "$unit"; then
    systemctl is-active "$unit"
  else
    echo "sudo restart failed — run: sudo systemctl restart $unit" >&2
    exit 2
  fi
else
  echo "No llamagate systemd unit found. Restart the process yourself." >&2
  exit 2
fi

echo "Verify (expect tokens_total to climb on a non-stream /v1/chat/completions):"
echo "  before=\$(curl -s http://127.0.0.1:11434/proxy/stats)"
echo "  curl -s http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"gpt-oss:120b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}' >/dev/null"
echo "  curl -s http://127.0.0.1:11434/proxy/stats"
