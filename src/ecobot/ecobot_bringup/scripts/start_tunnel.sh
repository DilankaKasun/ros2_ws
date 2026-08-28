#!/bin/bash
# ==============================================================================
# EcoBot Cloudflare Tunnel Auto-Connector
# Starts Cloudflare Tunnel for Rosbridge & Web Dashboard and updates dashboard config
# ==============================================================================

set -e

CLOUDFLARED_BIN="$(which cloudflared 2>/dev/null || echo "$HOME/.local/bin/cloudflared")"

if [ ! -x "$CLOUDFLARED_BIN" ]; then
  echo "Error: cloudflared binary not found at $CLOUDFLARED_BIN"
  exit 1
fi

PORT="${1:-9090}"
WEB_DIR="$HOME/remote_web_ui"
LOG_FILE="/tmp/cloudflared_${PORT}.log"
CONFIG_FILE="$WEB_DIR/tunnel_config.json"

echo "=== Starting Cloudflare Tunnel for Port $PORT ==="

# Kill existing tunnel for this port
pkill -f "cloudflared.*:${PORT}" 2>/dev/null || true
rm -f "$LOG_FILE"

# Start cloudflared in background and pipe output
"$CLOUDFLARED_BIN" tunnel --url "http://localhost:${PORT}" > "$LOG_FILE" 2>&1 &
TUNNEL_PID=$!

echo "Waiting for Cloudflare tunnel URL to be assigned..."

TUNNEL_URL=""
for i in {1..30}; do
  if [ -f "$LOG_FILE" ]; then
    TUNNEL_URL=$(grep -o "https://[a-zA-Z0-9.-]*\.trycloudflare\.com" "$LOG_FILE" | head -n 1 || true)
    if [ -n "$TUNNEL_URL" ]; then
      break
    fi
  fi
  sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
  echo "Error: Failed to obtain Cloudflare tunnel URL after 30s."
  echo "Log output:"
  cat "$LOG_FILE"
  exit 1
fi

WSS_URL=$(echo "$TUNNEL_URL" | sed 's/https:/wss:/')

echo ""
echo "=============================================================================="
echo "                   CLOUDFLARE TUNNEL CONNECTED SUCCESSFULLY                   "
echo "=============================================================================="
echo "  Public HTTPS URL:   $TUNNEL_URL"
echo "  WebSocket (WSS):    $WSS_URL"
echo "  Target Port:        http://localhost:$PORT"
echo "  Process PID:        $TUNNEL_PID"
echo "=============================================================================="

# Write config file for web UI auto-discovery
mkdir -p "$WEB_DIR"
cat <<EOF > "$CONFIG_FILE"
{
  "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "port": $PORT,
  "tunnel_https": "$TUNNEL_URL",
  "tunnel_wss": "$WSS_URL"
}
EOF

TUNNEL_HOST=$(echo "$TUNNEL_URL" | sed 's~https://~~')

echo "  Updated config:     $CONFIG_FILE"
echo ""
echo "  Direct Dashboard Links:"
echo "  [1] Vercel Hosted UI:  https://ecobot-ui.vercel.app/?robot=$TUNNEL_HOST"
echo "  [2] Local Web UI:      http://localhost:8080/?tunnel=$TUNNEL_URL"
echo "=============================================================================="
echo "Press Ctrl+C to stop tunnel."

trap "echo 'Stopping tunnel...'; kill $TUNNEL_PID 2>/dev/null; exit 0" INT TERM
wait $TUNNEL_PID
