#!/bin/bash
# Runs under tini (PID 1) as the container's entrypoint. Starts nginx (the
# single published-port gateway) and Streamlit, then waits for either to
# exit — if one dies, this script exits too, so the container exits and
# `restart: unless-stopped` brings both back up together rather than leaving
# a half-broken container (e.g. nginx dead but Streamlit still running,
# silently unreachable from outside).
set -e

if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY must be set — it protects the /api/* routes exposed" >&2
    echo "       through the single published port. Example:" >&2
    echo "         API_KEY=\$(openssl rand -hex 32) docker compose up" >&2
    exit 1
fi

sed "s/__API_KEY__/${API_KEY}/g" /app/nginx.conf.template > /etc/nginx/nginx.conf
nginx -t
nginx
NGINX_PID="$(cat /run/nginx.pid)"

/app/.venv-app/bin/python -m streamlit run /app/app.py \
    --server.port=8501 --server.address=127.0.0.1 --server.headless=true &
STREAMLIT_PID=$!

wait -n "$STREAMLIT_PID" "$NGINX_PID"
echo "nginx or streamlit exited — stopping the container so it restarts cleanly." >&2
kill "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null
exit 1
