#!/usr/bin/env bash
# Start Label Studio, self-contained in this folder, serving images/ as local files.
#
#   ./run.sh                                   # local: http://localhost:8080
#   ./run.sh https://abc123.ngrok-free.app     # expose via a tunnel (ngrok/cloudflared)
#
# Pass the PUBLIC https URL when tunnelling so Django trusts it for CSRF (else the
# login POST is rejected with "Forbidden (403) CSRF verification failed").
#
# Everything (SQLite DB, media) is kept under .label_studio/ so this folder is
# fully self-contained and nothing leaks into your home dir.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$DIR/images"
# Label Studio forbids pointing storage at the document root itself, so images
# live in a subfolder and the storage connection points here.
LABEL_DIR="$IMAGES_DIR/to_label"

# Optional public URL (1st arg or $LS_PUBLIC_URL) for tunnelled access. This LS
# version only trusts origins from the explicit CSRF_TRUSTED_ORIGINS env var, so
# set both that and the host to the tunnel's https URL.
PUBLIC_URL="${1:-${LS_PUBLIC_URL:-}}"
if [ -n "$PUBLIC_URL" ]; then
  export LABEL_STUDIO_HOST="$PUBLIC_URL"
  export CSRF_TRUSTED_ORIGINS="$PUBLIC_URL"
fi

# Keep all Label Studio state inside this project folder.
export LABEL_STUDIO_BASE_DATA_DIR="$DIR/.label_studio"

# Allow serving images straight from the local images/ folder (both env-var
# spellings are set so this works across Label Studio versions).
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$IMAGES_DIR"
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT="$IMAGES_DIR"

mkdir -p "$LABEL_DIR" "$LABEL_STUDIO_BASE_DATA_DIR"

# Persist a stable SECRET_KEY (kept in the git-ignored data dir) so login
# sessions survive restarts and Label Studio stops warning about a random key.
SECRET_FILE="$LABEL_STUDIO_BASE_DATA_DIR/.secret_key"
if [ ! -f "$SECRET_FILE" ]; then
  python -c "import secrets; print(secrets.token_urlsafe(50))" > "$SECRET_FILE"
fi
export SECRET_KEY="$(cat "$SECRET_FILE")"

# Preset login: Label Studio auto-creates this account on startup if missing, so
# there's no signup step. Override by exporting these before calling run.sh.
# Local-only tool (binds to localhost) — fine to keep simple creds here.
export LABEL_STUDIO_USERNAME="${LABEL_STUDIO_USERNAME:-admin@miners.local}"
export LABEL_STUDIO_PASSWORD="${LABEL_STUDIO_PASSWORD:-MinersLabel2026}"

echo "Login (preset)         : $LABEL_STUDIO_USERNAME  /  $LABEL_STUDIO_PASSWORD"
[ -n "$PUBLIC_URL" ] && echo "Public URL (CSRF ok)   : $PUBLIC_URL"
echo "Label Studio data dir : $LABEL_STUDIO_BASE_DATA_DIR"
echo "Document root          : $IMAGES_DIR"
echo "Put images to label in : $LABEL_DIR"
echo "When adding Source Storage in the UI, use this absolute path:"
echo "    $LABEL_DIR"
echo

exec label-studio start
