"""Sign catalogue and rule definitions (``config/rules/signs.json`` and ``rules.json``, CLAUDE.md §11).

The catalogue says what each sign code is called, its shape (for the relevance check) and which
voice prompt belongs to it; ``rules.json`` says what a sign does (set a speed limit, ban a turn...).
Official Russian texts are filled in from ПДД РФ and verified by a human (M2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parents[1] / "config" / "rules"


@dataclass(frozen=True)
class Catalogue:
    signs: dict[str, dict]     # code -> catalogue entry
    rules: dict                # rules.json

    @classmethod
    def load(cls, rules_dir: Path = RULES_DIR) -> "Catalogue":
        signs = json.loads((rules_dir / "signs.json").read_text(encoding="utf-8"))["signs"]
        rules = json.loads((rules_dir / "rules.json").read_text(encoding="utf-8"))
        return cls(signs={s["code"]: s for s in signs}, rules=rules)

    def shape(self, code: str) -> str:
        return self.signs.get(code, {}).get("shape", "round")

    def is_plate(self, code: str) -> bool:
        return code.startswith("8.")

    def label(self, code: str, value: int | None) -> str:
        """Short English label for overlays, e.g. "Maximum speed limit 60"."""
        name = self.signs.get(code, {}).get("name", code)
        return f"{name} {value}" if value is not None else name

    def voice_key(self, code: str, value: int | None) -> str | None:
        key = self.signs.get(code, {}).get("voice_key")
        if key is None or ("{value}" in key and value is None):
            return None
        return key.format(value=value)

    def rule(self, code: str) -> dict | None:
        rule = self.rules["signs"].get(code)
        return rule if rule and rule.get("enabled", True) else None

    def explainable_codes(self, detectable: set[str]) -> list[str]:
        """Catalogue signs (not plates) the active detector can see — basis for "signs not met yet"."""
        return sorted(c for c in self.signs if c in detectable and not self.is_plate(c))
