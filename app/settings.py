"""Loads configuration: ``config/settings.yaml`` plus a few environment variables (or ``.env``).

Why a separate module: every part of the app and pipeline needs the same thresholds and paths,
and thresholds must live in config, not in code (CLAUDE.md §3). Paths are split in two:

- the shared data root (``SIGNCOACH_DATA``): datasets, model weights, training runs;
- an instance state folder (``SIGNCOACH_INSTANCE``): database, uploads, jobs and media. The
  deployed site uses ``prod``; a developer server uses ``dev``; tests use their own temp folder.
  Separate state means a dev server can never pick up production jobs from the shared queue.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "config"


@dataclass(frozen=True)
class Settings:
    cfg: dict          # everything from config/settings.yaml (thresholds)
    data_dir: Path     # shared: datasets, models, training runs
    instance: str      # prod | dev | test | ...
    device: str        # cpu | cuda:0
    git_sha: str

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "instances" / self.instance

    @property
    def db_path(self) -> Path:
        return self.state_dir / "db" / "signcoach.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.state_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.state_dir / "jobs"

    @property
    def media_dir(self) -> Path:
        return self.state_dir / "media"

    @property
    def models_dir(self) -> Path:
        # Tests use a throw-away data folder but the real weights, hence the separate override.
        return Path(os.environ.get("SIGNCOACH_MODELS") or self.data_dir / "models").expanduser()

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.uploads_dir, self.jobs_dir, self.media_dir):
            d.mkdir(parents=True, exist_ok=True)


def _git_sha() -> str:
    """The deployed commit, shown by /health as deployment evidence."""
    if os.environ.get("SIGNCOACH_GIT_SHA"):
        return os.environ["SIGNCOACH_GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(REPO_DIR / ".env", override=False)  # real environment variables win over .env
    cfg = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    device = os.environ.get("SIGNCOACH_DEVICE") or cfg.get("device", "cuda:0")
    cfg["device"] = device
    return Settings(
        cfg=cfg,
        data_dir=Path(os.environ.get("SIGNCOACH_DATA", "~/signcoach-data")).expanduser(),
        instance=os.environ.get("SIGNCOACH_INSTANCE", "dev"),
        device=device,
        git_sha=_git_sha(),
    )
