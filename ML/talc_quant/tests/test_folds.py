"""Grouped-fold correctness: specimen grouping + no-straddle guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.folds import assign_folds, base_token, specimen_map  # noqa: E402


def test_sample_number_is_specimen():
    assert base_token("2550374-2_10.jpg") == "2550374"
    assert base_token("2550374-1 5x.JPG") == "2550374"
    # same sample number, different subsample/zoom -> same specimen
    names = ["2550374-2_10.jpg", "2550374-1 5x.JPG", "2550375-3_5.jpg"]
    sm = specimen_map(names)
    assert sm["2550374-2_10.jpg"] == sm["2550374-1 5x.JPG"]
    assert sm["2550375-3_5.jpg"] != sm["2550374-2_10.jpg"]


def test_dscn_gap_clustering():
    names = ["DSCN3048.jpg", "DSCN3051.jpg", "DSCN3052.jpg",
             "DSCN4714.jpg", "DSCN4715.jpg"]
    sm = specimen_map(names)
    # tight numeric run -> one specimen
    assert sm["DSCN3048.jpg"] == sm["DSCN3051.jpg"] == sm["DSCN3052.jpg"]
    assert sm["DSCN4714.jpg"] == sm["DSCN4715.jpg"]
    # big gap -> different specimen
    assert sm["DSCN3048.jpg"] != sm["DSCN4714.jpg"]


def test_no_specimen_straddles_folds():
    names = [f"2550{n}-{s}_10.jpg" for n in range(370, 380) for s in (1, 2)]
    names += [f"DSCN{n}.jpg" for n in list(range(3040, 3060)) + list(range(4700, 4740))]
    folds = assign_folds(names, k=5, seed=42)
    sm = specimen_map(names)
    spec_to_folds: dict[str, set[int]] = {}
    for name, f in folds.items():
        spec_to_folds.setdefault(sm[name], set()).add(f)
    for spec, fset in spec_to_folds.items():
        assert len(fset) == 1, f"specimen {spec} split across folds {fset}"


def test_assign_folds_deterministic():
    names = [f"DSCN{n}.jpg" for n in range(3000, 3100, 3)]
    assert assign_folds(names, 5, 42) == assign_folds(names, 5, 42)


def test_folds_cover_all_indices():
    names = [f"2550{n}-1_10.jpg" for n in range(300, 340)]
    folds = assign_folds(names, k=5, seed=1)
    assert set(folds.values()) == set(range(5))
