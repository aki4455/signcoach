#!/usr/bin/env bash
# One-time host setup that needs root (CLAUDE.md §16). Run it yourself in a WSL terminal:
#     sudo bash deploy/setup_host.sh
# Safe to run again. Everything else (repo, deploy checkout, runner registration) needs no root.
#
# It does four things:
#   1. installs Tailscale (for the public HTTPS URL via Funnel) if missing
#   2. installs the sudoers rule that lets the deploy workflow run `systemctl restart signcoach`
#   3. installs and starts the signcoach systemd service
#   4. installs the GitHub Actions runner as a service, if the runner was already configured
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then echo "Run with sudo: sudo bash $0" >&2; exit 1; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="akmal"
HOME_DIR="/home/$APP_USER"

echo "== 1/4 Tailscale"
if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
systemctl enable --now tailscaled

echo "== 2/4 sudoers rule"
visudo -cf "$REPO/deploy/sudoers.example"
install -m 0440 -o root -g root "$REPO/deploy/sudoers.example" /etc/sudoers.d/signcoach

echo "== 3/4 signcoach service"
install -m 0644 "$REPO/deploy/signcoach.service" /etc/systemd/system/signcoach.service
systemctl daemon-reload
systemctl enable signcoach
if [ -x "$HOME_DIR/signcoach-deploy/.venv/bin/uvicorn" ]; then
  systemctl restart signcoach
  sleep 3
  curl -fsS http://127.0.0.1:8000/health && echo
else
  echo "   (deploy checkout not ready yet; the service will start after the first deploy)"
fi

echo "== 4/4 GitHub Actions runner service"
if [ -f "$HOME_DIR/actions-runner/.runner" ]; then
  cd "$HOME_DIR/actions-runner"
  ./svc.sh status >/dev/null 2>&1 || ./svc.sh install "$APP_USER"
  ./svc.sh start || true
else
  echo "   (runner not configured yet — skip; run this script again after configuring it)"
fi

cat <<'EOF'

Next steps (see deploy/FUNNEL.md):
  sudo tailscale up --hostname=signcoach        # opens a login link
  sudo tailscale funnel --bg 8000               # public HTTPS -> 127.0.0.1:8000
  tailscale funnel status                       # shows the public URL
EOF
