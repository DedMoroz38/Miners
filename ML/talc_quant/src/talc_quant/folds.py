"""Grouped cross-validation by specimen — a specimen never straddles folds.

Two naming families:
  * "2550374-2 10х"  -> specimen = the leading sample number "2550374"
    (the "-2" is a subsample/field of the same polished section; "10х" is zoom).
  * "DSCN3048"       -> Canon sequential frames; consecutive numbers are the
    same specimen, a large gap starts a new one. We cluster by numeric gap.
Anything else (e.g. part2 negatives "-1", "-10") becomes its own token so the
negatives spread evenly across folds.
"""
from __future__ import annotations

import re
from pathlib import Path

_SAMPLE_RE = re.compile(r"^(\d{5,})")
_DSCN_RE = re.compile(r"^dscn[_-]?(\d+)", re.IGNORECASE)

# Consecutive DSCN frames within this numeric gap belong to the same specimen.
DSCN_GAP = 10


def base_token(name: str) -> str:
    """Per-name coarse token (pre-clustering). DSCN keeps its number for later."""
    stem = Path(name).stem
    m = _SAMPLE_RE.match(stem)
    if m:
        return m.group(1)
    m = _DSCN_RE.match(stem)
    if m:
        return f"DSCN{int(m.group(1))}"
    return stem


def _dscn_number(token: str) -> int | None:
    return int(token[4:]) if token.startswith("DSCN") else None


def specimen_map(names: list[str]) -> dict[str, str]:
    """Map each name -> a specimen key, clustering DSCN frames by numeric gap."""
    tokens = {n: base_token(n) for n in names}

    dscn_nums = sorted({_dscn_number(t) for t in tokens.values()
                        if _dscn_number(t) is not None})
    num_to_cluster: dict[int, str] = {}
    if dscn_nums:
        cluster_idx = 0
        num_to_cluster[dscn_nums[0]] = f"DSCN_c{cluster_idx}"
        for prev, cur in zip(dscn_nums, dscn_nums[1:]):
            if cur - prev > DSCN_GAP:
                cluster_idx += 1
            num_to_cluster[cur] = f"DSCN_c{cluster_idx}"

    out: dict[str, str] = {}
    for name, tok in tokens.items():
        num = _dscn_number(tok)
        out[name] = num_to_cluster[num] if num is not None else tok
    return out


def specimen_id(name: str, names: list[str] | None = None) -> str:
    """Specimen key for one name. Pass `names` for correct DSCN clustering."""
    if names is None:
        names = [name]
    return specimen_map(names)[name]


def assign_split(names: list[str], test_frac: float = 0.2,
                 seed: int = 42) -> dict[str, str]:
    """Grouped train/test split (no k-fold): every specimen goes entirely to
    "train" or "test", and test holds ≈ test_frac of the IMAGES. Deterministic
    given (names, test_frac, seed). A specimen never straddles the split."""
    import random

    spec = specimen_map(names)
    counts: dict[str, int] = {}
    for n in names:
        counts[spec[n]] = counts.get(spec[n], 0) + 1

    total = len(names)
    target_test = max(1, round(test_frac * total)) if total else 0
    cap = max(1, round(target_test * 1.5))           # never let test exceed this

    specimens = sorted(counts)                       # deterministic order
    rng = random.Random(seed)
    rng.shuffle(specimens)
    # Greedily fill test in shuffled order, skipping any specimen that would push
    # test past the cap. A single dominant specimen (e.g. a big DSCN cluster that
    # is ~40% of the data) is thus skipped and stays in TRAIN rather than
    # swallowing the whole test set.
    test_specs: set[str] = set()
    filled = 0
    for s in specimens:
        if filled >= target_test:
            break
        if filled + counts[s] <= cap:
            test_specs.add(s)
            filled += counts[s]
    if not test_specs and specimens:                 # guarantee a non-empty test
        smallest = min(specimens, key=lambda x: counts[x])
        test_specs.add(smallest)

    return {n: ("test" if spec[n] in test_specs else "train") for n in names}
