# Deployment and public URL (Tailscale Funnel)

How the deployed site works (CLAUDE.md §16):

- The app runs as the systemd service `signcoach` from a separate checkout, `~/signcoach-deploy`
  (never from the development checkout), on `127.0.0.1:8000`.
- A push to `main` triggers `.github/workflows/deploy.yml` on our self-hosted runner. The workflow
  checks out the commit in `~/signcoach-deploy`, runs `uv sync --frozen`, builds the frontend, runs the
  fast tests, restarts the service, and waits until `/health` reports the new git SHA.
- Tailscale Funnel publishes `127.0.0.1:8000` as `https://signcoach.<tailnet>.ts.net`.

## One-time setup

1. **GitHub CLI login** (lets Claude create the repo, open PRs and register the runner):
   `gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key` — log in as the
   account whose SSH key is on this machine.
2. **Runner registration** (no root): Claude downloads the runner to `~/actions-runner` and runs
   `./config.sh --url https://github.com/<owner>/signcoach --token <token> --name signcoach --labels signcoach --unattended`.
   The token comes from `gh api -X POST repos/<owner>/signcoach/actions/runners/registration-token`.
3. **Root steps** — run yourself in a WSL terminal: `sudo bash deploy/setup_host.sh`
   (installs Tailscale, the sudoers rule, the `signcoach` service and the runner service).
4. **Tailscale login:** `sudo tailscale up --hostname=signcoach` and open the printed link.
5. **Enable Funnel** for the tailnet: in the admin console (https://login.tailscale.com/admin) enable
   *HTTPS certificates* (DNS page) and allow Funnel for this machine (the next command prints a link
   if it is not allowed yet).
6. **Publish:** `sudo tailscale funnel --bg 8000`, then `tailscale funnel status` shows the URL.
   The first certificate can take up to ~10 minutes.
7. **Test from a phone on mobile data** (not on the same Wi-Fi): open the URL and `/health`.

## Keep it online (the site is monitored after the deadline)

- Windows: power plan → never sleep / never hibernate; pause Windows Update until after the event.
- WSL can stop the Ubuntu VM when no WSL terminal is open, which takes the site down. Keep one WSL
  terminal open on the host at all times (e.g. the `tmux` session). Check: close every other
  terminal, wait two minutes, open the public URL from a phone.
- After a reboot: open a WSL terminal; systemd starts `signcoach`, `tailscaled` and the runner.

## Security notes

- The repository is public and the runner executes code on this machine. The workflow runs only on
  pushes to `main` and manual dispatch — never on pull requests. In the repo settings set
  *Actions → General → Fork pull request workflows from outside collaborators → Require approval for
  all outside collaborators*, and do not merge pull requests from forks during the event.
- The sudoers rule allows exactly `systemctl restart signcoach`, nothing else.

## Freeze

At the deadline tag the submitted commit `final-submission`. After that, only redeploy that tag:
*Actions → deploy → Run workflow → ref = final-submission*.
