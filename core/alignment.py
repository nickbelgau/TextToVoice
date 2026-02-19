# core/alignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import difflib


def _try_import_rapidfuzz():
    try:
        from rapidfuzz import fuzz  # type: ignore
        return fuzz
    except Exception:
        return None


_RAPIDFUZZ = _try_import_rapidfuzz()


def normalize_for_match(s: str) -> str:
    """
    Normalize text for matching:
    - lower
    - remove hyphen line breaks "-\n"
    - collapse whitespace to single spaces
    - replace punctuation with spaces
    - keep only alnum + spaces
    """
    if not s:
        return ""

    out = []
    prev_space = True
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # remove PDF-style hyphenation across line breaks: "-\n", "-\r\n"
        if ch == "-" and i + 1 < n and s[i + 1] in ("\n", "\r"):
            i += 1
            # skip newlines + surrounding whitespace
            while i < n and s[i] in ("\r", "\n", " ", "\t"):
                i += 1
            continue

        low = ch.lower()

        if low.isalnum():
            out.append(low)
            prev_space = False
        elif low.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            # punctuation -> space (but collapse)
            if not prev_space:
                out.append(" ")
                prev_space = True

        i += 1

    # strip leading/trailing spaces
    return "".join(out).strip()


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """
    Returns (norm_string, norm_index_to_orig_index)
    norm_index_to_orig_index[i] gives the original char index in `s` that produced norm[i].
    """
    if not s:
        return "", []

    out_chars: list[str] = []
    idx_map: list[int] = []

    prev_space = True
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # remove hyphenation "-\n"
        if ch == "-" and i + 1 < n and s[i + 1] in ("\n", "\r"):
            i += 1
            while i < n and s[i] in ("\r", "\n", " ", "\t"):
                i += 1
            continue

        low = ch.lower()

        if low.isalnum():
            out_chars.append(low)
            idx_map.append(i)
            prev_space = False
        elif low.isspace():
            if not prev_space:
                out_chars.append(" ")
                idx_map.append(i)
                prev_space = True
        else:
            if not prev_space:
                out_chars.append(" ")
                idx_map.append(i)
                prev_space = True

        i += 1

    # trim leading/trailing space in norm, while keeping map aligned
    if not out_chars:
        return "", []

    # trim left
    while out_chars and out_chars[0] == " ":
        out_chars.pop(0)
        idx_map.pop(0)

    # trim right
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        idx_map.pop()

    return "".join(out_chars), idx_map


def _score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _RAPIDFUZZ is not None:
        return int(_RAPIDFUZZ.ratio(a, b))
    return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)


@dataclass
class AlignConfig:
    lookback: int = 250
    ahead: int = 9000
    threshold: int = 78
    min_query_len: int = 10


def find_best_match(
    norm_source: str,
    norm_to_orig: list[int],
    query_norm: str,
    cursor: int,
    cfg: AlignConfig,
) -> Optional[tuple[int, int, int, int]]:
    """
    Returns (orig_start, orig_end, new_cursor_norm, score) in SOURCE coordinates,
    or None if no match.
    """
    if not query_norm or len(query_norm) < cfg.min_query_len:
        return None

    n = len(norm_source)
    L = len(query_norm)

    start_search = max(0, cursor - cfg.lookback)
    end_search = min(n, cursor + cfg.ahead)

    # Phase A: exact find in the local window
    idx = norm_source.find(query_norm, start_search, end_search)
    if idx != -1:
        norm_s = idx
        norm_e = idx + L
        if norm_e - 1 < len(norm_to_orig):
            orig_s = norm_to_orig[norm_s]
            orig_e = norm_to_orig[norm_e - 1] + 1
            return orig_s, orig_e, norm_e, 100

    # Phase B: fuzzy window scan in the local window
    region_start = start_search
    region_end = end_search
    if region_end - region_start < L:
        return None

    step = max(1, L // 6)
    best_score = -1
    best_norm_s = -1

    # compare fixed-length windows of length L (fast + stable)
    for norm_s in range(region_start, region_end - L + 1, step):
        cand = norm_source[norm_s : norm_s + L]
        sc = _score(query_norm, cand)
        if sc > best_score:
            best_score = sc
            best_norm_s = norm_s
            if best_score >= 96:  # early exit if basically perfect
                break

    if best_score < cfg.threshold or best_norm_s < 0:
        return None

    norm_s = best_norm_s
    norm_e = norm_s + L
    orig_s = norm_to_orig[norm_s]
    orig_e = norm_to_orig[norm_e - 1] + 1
    return orig_s, orig_e, norm_e, int(best_score)


def align_segments_to_text(
    source_text: str,
    segments: list[dict],
    cfg: AlignConfig | None = None,
) -> list[dict]:
    """
    Adds:
      - orig_char_start_local
      - orig_char_end_local
      - align_score
    to segment dicts (when matched). Returns same list (mutated) for convenience.

    Monotonic alignment: cursor only moves forward.
    """
    if cfg is None:
        cfg = AlignConfig()

    norm_source, norm_map = normalize_with_map(source_text)
    if not norm_source:
        return segments

    cursor = 0
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        q = normalize_for_match(txt)

        m = find_best_match(norm_source, norm_map, q, cursor, cfg)
        if m is None:
            # avoid getting stuck forever: advance slightly
            cursor = min(len(norm_source), cursor + max(1, len(q) // 3))
            continue

        local_s, local_e, new_cursor, score = m
        seg["orig_char_start_local"] = int(local_s)
        seg["orig_char_end_local"] = int(local_e)
        seg["align_score"] = int(score)
        cursor = new_cursor

    return segments
