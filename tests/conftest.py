"""Shared pytest setup: every test run gets its own throw-away data folder.

Tests must never touch the production database or media, so SIGNCOACH_DATA points to a temp
folder and SIGNCOACH_INSTANCE is "test". The device is CPU (the GPU may be training).
"""

from __future__ import annotations

import os
import tempfile

os.environ["SIGNCOACH_DATA"] = tempfile.mkdtemp(prefix="signcoach-test-")
os.environ["SIGNCOACH_INSTANCE"] = "test"
os.environ["SIGNCOACH_DEVICE"] = "cpu"
