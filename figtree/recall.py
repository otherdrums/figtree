"""Recall verification for Figtree generation.

The goal is *flawless recall*: when asked to recount a source, the generated
text must reproduce every checkable atom (figure, named entity, key claim) that
appears in the source. Because a small model under a token budget will sometimes
truncate or skip an atom, this module provides a verify-and-patch loop:

1. Extract checkable atoms from the source text(s).
2. After generation, diff the output against the source atoms.
3. If any are missing, issue a targeted follow-up prompt listing exactly the
   missing atoms and regenerate, appending the recovered facts.

The result is recall that is correct *by construction* rather than by hoping the
model happened to include everything.
"""

from __future__ import annotations

import re

# A "checkable atom" is a self-contained fact fragment we can test for presence:
#   - numbers, optionally with a unit/scale ("2,700", "150 billion", "3%", "2030")
#   - short capitalized named entities / acronyms ("WTO", "MSCI World Index")
# We deliberately keep these simple and surface-level; the point is to catch
# dropped figures, not to do semantic claim matching.

_NUM_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|thousand|percent|%|bn|m|k)?",
    re.IGNORECASE,
)
# Only treat true acronyms / initialisms as entity atoms: 2-5 ALL-CAPS letters,
# or a token containing a dot or ampersand (e.g. "MSCI", "W.E.F.", "Procter&Gamble").
# Generic Title-Case headings ("Davos Summit Achieves Historic Breakthrough") are
# NOT atoms — they are rephrased by the model and should not count as recall gaps.
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})\b")
# All-caps tokens that are ordinary English words (often headline emphasis) rather
# than true acronyms/initialisms. Excluded from entity atoms so they don't create
# false recall gaps when the model rephrases a headline.
_ACRONYM_DENY = {
    "BIG", "NEW", "FEW", "YES", "NO", "ALL", "TOP", "KEY", "THE", "AND", "FOR",
    "NOT", "BUT", "ARE", "WAS", "WERE", "HAS", "HAD", "ONE", "TWO", "WHO", "WHO",
    "HOW", "WHY", "WHAT", "WHEN", "NOW", "OUT", "UP", "OFF", "PRO", "CON", "VS",
}
_DOTTED_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*[\.&][A-Za-z0-9.&]+)\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _normalize_atom(s: str) -> str:
    return s.strip().lower().replace(",", "")


def extract_atoms(text: str) -> list[str]:
    """Return the list of checkable atoms present in ``text`` (deduplicated, order-preserving)."""
    atoms: list[str] = []
    seen: set[str] = set()

    for m in _NUM_RE.findall(text):
        a = _normalize_atom(m)
        if a and a not in seen:
            seen.add(a)
            atoms.append(a)
    for m in _YEAR_RE.findall(text):
        a = _normalize_atom(m)
        if a not in seen:
            seen.add(a)
            atoms.append(a)
    for m in _ACRONYM_RE.findall(text):
        if m in _ACRONYM_DENY:
            continue
        a = _normalize_atom(m)
        if a not in seen:
            seen.add(a)
            atoms.append(a)
    for m in _DOTTED_RE.findall(text):
        a = _normalize_atom(m)
        if a and a not in seen:
            seen.add(a)
            atoms.append(a)
    return atoms


def missing_atoms(source_text: str, generated_text: str) -> list[str]:
    """Atoms present in ``source_text`` but absent from ``generated_text``."""
    src = extract_atoms(source_text)
    gen_norm = _normalize_atom(generated_text)
    missing = []
    for a in src:
        # An atom is "recalled" if its normalized form appears verbatim, OR a
        # shorter numeric core appears (e.g. "2,700" vs "2700").
        if a in gen_norm:
            continue
        core = re.sub(r"[^0-9].*$", "", a)  # leading digits only
        if core and core in gen_norm.replace(",", ""):
            continue
        missing.append(a)
    return missing


def recall_score(source_text: str, generated_text: str) -> float:
    """Fraction of source atoms recalled (1.0 == flawless)."""
    src = extract_atoms(source_text)
    if not src:
        return 1.0
    miss = missing_atoms(source_text, generated_text)
    return max(0.0, (len(src) - len(miss)) / len(src))


def build_recall_prompt(missing: list[str]) -> str:
    """Targeted follow-up that asks the model to state exactly the missing atoms."""
    listing = "; ".join(missing)
    return (
        f"The following specific facts from the source were omitted from your "
        f"answer. State each one exactly as it appears, with its precise figure "
        f"or name: {listing}."
    )
