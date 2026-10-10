"""Copy the official texts of the Russian Road Traffic Rules (ПДД РФ) into the catalogue and rules
(CLAUDE.md hard rule 2, §11.2), and write the human checklist ``docs/RULES_CHECKLIST.md``.

Why a script: legal texts must never be invented or paraphrased. This copies them verbatim from the
official text (ConsultantPlus copy of Government Decree No. 1090, current edition), records where
each one came from, and leaves every entry ``"verified": false`` until a Russian-speaking team
member has checked it against the source and flipped it by hand.

What it copies:
- signs.json  -> for every sign: ``official.name_ru``, ``official.meaning_ru`` (the definition
  paragraph after the name) and ``official.notes_ru`` (the paragraphs that directly follow it);
- rules.json  -> the zone-of-action sentences of each sign ("Действие знаков ... распространяется",
  "Зона действия знаков ..."), and the default speed limits of section 10.2/10.3 with the numbers
  parsed from those sentences.
Amendment notes like "(в ред. Постановления ...)" are dropped; nothing else is changed.

    uv run python scripts/extract_pdd.py             # download pages, update the JSON files, write the checklist
    uv run python scripts/extract_pdd.py --cache DIR # reuse/save the downloaded pages in DIR
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIGNS_JSON = REPO / "config" / "rules" / "signs.json"
RULES_JSON = REPO / "config" / "rules" / "rules.json"
CHECKLIST = REPO / "docs" / "RULES_CHECKLIST.md"
SPEED_URL = "https://www.consultant.ru/document/cons_doc_LAW_2709/803907f6876a655f77995888208488d4bfccf247/"
CODE = r"\d+\.\d+(?:\.\d+)?"
# Paragraphs that are amendment history, not rule text.
NOTE = re.compile(r"^\((в ред\.|см\. текст|абзац введен|примечание|п\. \d|введен)|^Абзац.*(утратил силу|исключен)|"
                  r"^абзац исключен|^Абзацы .* исключены", re.IGNORECASE)


class _Paragraphs(HTMLParser):
    """HTML -> list of text paragraphs (whitespace normalised)."""

    BLOCK = {"p", "div", "br", "li", "td", "h1", "h2", "h3"}

    def __init__(self):
        super().__init__()
        self.paras, self._buf, self._skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in ("script", "style")
        if tag in self.BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        self._skip -= tag in ("script", "style")
        if tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self.paras.append(text)


def fetch(url: str, cache: Path | None) -> str:
    name = url.rstrip("/").split("/")[-1] + ".html"
    if cache and (cache / name).exists():
        return (cache / name).read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SignCoach rules extraction)"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(html, encoding="utf-8")
    time.sleep(1)  # be polite to the site
    return html


def paragraphs(html: str) -> list[str]:
    p = _Paragraphs()
    p.feed(html)
    p._flush()
    return [t for t in p.paras if not NOTE.search(t)]


def expand(codes_text: str) -> set[str]:
    """'5.19.1, 5.19.2' / '8.22.1 - 8.22.3' / '3.26 - 3.30' -> set of codes."""
    out: set[str] = set()
    for part in re.split(r",|\sи\s", codes_text):
        m = re.match(rf"\s*({CODE})\s*(?:-\s*({CODE}))?\s*$", part)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        out.add(a)
        if b:
            pa, pb = a.split("."), b.split(".")
            if pa[:-1] == pb[:-1]:  # same prefix: count the last number
                out.update(".".join(pa[:-1] + [str(n)]) for n in range(int(pa[-1]), int(pb[-1]) + 1))
            else:
                out.add(b)
    return out


def definition(code: str, paras: list[str]) -> dict | None:
    """Find the paragraph that defines `code` and the paragraphs that belong to it."""
    named = re.compile(rf"((?:{CODE})(?:\s*(?:,|-|и)\s*{CODE})*)\s*\"([^\"]+)\"")
    for i, text in enumerate(paras):
        if not re.match(CODE, text):
            continue
        matches = list(named.finditer(text))
        hit = next((m for m in matches if code in expand(m.group(1))), None)
        if hit is None:
            continue
        meaning = text[matches[-1].end():].lstrip(".").strip()
        notes = []
        for follow in paras[i + 1:i + 4]:
            if re.match(CODE, follow) or follow.endswith(":"):
                break
            notes.append(follow)
        return {"name_ru": hit.group(2), "meaning_ru": meaning, "notes_ru": notes}
    return None


def scope_sentences(code: str, paras: list[str]) -> list[str]:
    """Zone-of-action sentences that mention `code` ('Действие знаков ...', 'Зона действия знаков ...')."""
    out = []
    for text in paras:
        m = re.match(rf"^(?:Действие|Зона действия) знак(?:а|ов) ((?:{CODE})(?:\s*(?:,|-|и)\s*{CODE})*)", text)
        if (m and code in expand(m.group(1))) or (text.startswith("для знаков") and code in expand(
                re.match(rf"для знаков ((?:{CODE})(?:\s*(?:,|-|и)\s*{CODE})*)", text).group(1))):
            out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, help="folder to keep downloaded pages")
    args = ap.parse_args()

    catalogue = json.loads(SIGNS_JSON.read_text(encoding="utf-8"))
    rules = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    pages: dict[str, list[str]] = {}
    editions = set()
    for entry in catalogue["signs"]:
        url = entry["source"]["url"]
        if url not in pages:
            html = fetch(url, args.cache)
            pages[url] = paragraphs(html)
            m = re.search(r"ред\. от (\d\d\.\d\d\.\d{4})", html)
            editions.add(m.group(1) if m else "?")
        found = definition(entry["code"], pages[url])
        if found is None:
            print(f"!! no definition found for {entry['code']}")
            continue
        entry["official"] = found
        entry["source"]["edition"] = f"ред. от {sorted(editions)[-1]}"
        rule = rules["signs"].get(entry["code"])
        if rule is not None:
            scope = scope_sentences(entry["code"], pages[url])
            if scope:
                rule.setdefault("scope", {})["official_ru"] = scope
        print(f"ok {entry['code']:7s} {found['name_ru']}")

    speed = paragraphs(fetch(SPEED_URL, args.cache))
    p102 = next(t for t in speed if t.startswith("10.2."))
    p103 = next(t for t in speed if t.startswith("мотоциклам, легковым автомобилям"))
    inside = int(re.search(r"не более (\d+) км/ч", p102).group(1))
    outside = int(re.search(r"на остальных дорогах - не более (\d+) км/ч", p103).group(1))
    rules["defaults"]["speed_limit_kmh"] = {"in_settlement": inside, "outside_settlement": outside}
    rules["defaults"]["official_ru"] = [p102, "10.3. Вне населенных пунктов разрешается движение: " + p103]
    rules["defaults"]["source"]["edition"] = f"ред. от {sorted(editions)[-1]}"
    rules["defaults"]["verified"] = False
    print(f"ok defaults: {inside} km/h in built-up areas, {outside} km/h outside (cars)")

    SIGNS_JSON.write_text(json.dumps(catalogue, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_checklist(catalogue, rules)


def write_checklist(catalogue: dict, rules: dict) -> None:
    lines = ["# Rules checklist — check every line against the official text", "",
             "Generated by `scripts/extract_pdd.py`. For each row: open the link, compare the Russian text word "
             "for word, check the English draft says the same, then set `\"verified\": true` for that entry in "
             "`config/rules/signs.json` (or `rules.json` for the defaults and zones of action).", "",
             "## Default speed limits (rules.json → defaults)", ""]
    d = rules["defaults"]
    lines += [f"- [ ] In built-up areas: **{d['speed_limit_kmh']['in_settlement']} km/h**, outside: "
              f"**{d['speed_limit_kmh']['outside_settlement']} km/h** (cars) — [{d['source']['section']}]({d['source']['url']})"]
    lines += [f"  > {t}" for t in d.get("official_ru", [])]
    lines += ["", "## Signs (signs.json)", ""]
    for e in catalogue["signs"]:
        o = e.get("official", {})
        lines += [f"### [ ] {e['code']} — {o.get('name_ru', 'TODO')} / {e['name']}",
                  f"Source: [{e['source']['section']}]({e['source']['url']}), {e['source']['edition']}", "",
                  f"> {o.get('meaning_ru') or '(the official text gives only the name)'}"]
        lines += [f"> {n}" for n in o.get("notes_ru", [])]
        lines += ["", f"English draft: {e['meaning']}"]
        scope = rules["signs"].get(e["code"], {}).get("scope", {}).get("official_ru", [])
        if scope:
            lines += ["", "Zone of action (rules.json):"] + [f"> {s}" for s in scope]
        lines += [""]
    CHECKLIST.parent.mkdir(parents=True, exist_ok=True)
    CHECKLIST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {CHECKLIST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
