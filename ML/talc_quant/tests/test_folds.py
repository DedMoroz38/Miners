"""Grouped train/test split: specimen grouping + no-straddle guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.folds import assign_split, base_token, specimen_map  # noqa: E402


def test_sample_number_is_specimen():
    assert base_token("2550374-2_10.jpg") == "2550374"
    assert base_token("2550374-1 5x.JPG") == "2550374"
    names = ["2550374-2_10.jpg", "2550374-1 5x.JPG", "2550375-3_5.jpg"]
    sm = specimen_map(names)
    assert sm["2550374-2_10.jpg"] == sm["2550374-1 5x.JPG"]
    assert sm["2550375-3_5.jpg"] != sm["2550374-2_10.jpg"]


def test_dscn_gap_clustering():
    names = ["DSCN3048.jpg", "DSCN3051.jpg", "DSCN3052.jpg",
             "DSCN4714.jpg", "DSCN4715.jpg"]
    sm = specimen_map(names)
    assert sm["DSCN3048.jpg"] == sm["DSCN3051.jpg"] == sm["DSCN3052.jpg"]
    assert sm["DSCN4714.jpg"] == sm["DSCN4715.jpg"]
    assert sm["DSCN3048.jpg"] != sm["DSCN4714.jpg"]


def test_no_specimen_straddles_split():
    names = [f"2550{n}-{s}_10.jpg" for n in range(370, 380) for s in (1, 2)]
    names += [f"DSCN{n}.jpg" for n in list(range(3040, 3060)) + list(range(4700, 4740))]
    split = assign_split(names, test_frac=0.2, seed=42)
    sm = specimen_map(names)
    spec_to_splits: dict[str, set[str]] = {}
    for name, s in split.items():
        spec_to_splits.setdefault(sm[name], set()).add(s)
    for spec, sset in spec_to_splits.items():
        assert len(sset) == 1, f"specimen {spec} split across {sset}"


def test_assign_split_deterministic():
    names = [f"DSCN{n}.jpg" for n in range(3000, 3100, 3)]
    assert assign_split(names, 0.2, 42) == assign_split(names, 0.2, 42)


def test_split_has_both_classes():
    names = [f"2550{n}-1_10.jpg" for n in range(300, 340)]
    split = assign_split(names, test_frac=0.25, seed=1)
    assert set(split.values()) == {"train", "test"}


def test_test_fraction_roughly_honored():
    names = [f"2550{n}-1_10.jpg" for n in range(300, 400)]   # 100 distinct specimens
    split = assign_split(names, test_frac=0.2, seed=3)
    n_test = sum(v == "test" for v in split.values())
    assert 10 <= n_test <= 30       # ~20 of 100, with grouped granularity slack
