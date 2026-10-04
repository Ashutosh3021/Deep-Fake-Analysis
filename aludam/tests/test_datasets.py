"""Tests for Phase 0 dataset loading.

The single most dangerous failure this guards against is a silently inverted
label convention: MAGE ships 1 = human, our convention is 1 = synthetic. An
inversion yields a full set of plausible-looking metrics that are exactly
backwards, and nothing downstream would notice.

Run:  python tests/test_datasets.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import datasets as D  # noqa: E402

PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


print("\n[1] manifest / dispatch")

check(os.path.exists(D.MANIFEST_PATH), "manifest exists (fetch.py has run)")
check(D.available_splits() == (
    "mage:test", "mage:valid", "mage:ood_gpt", "mage:ood_gpt_para", "hc3"),
    "five stable split names")

for bad in ("mage:", "mage:train", "imagenet", ""):
    try:
        D.load_split(bad)
        check(False, f"load_split({bad!r}) should raise")
    except D.DatasetError:
        check(True, f"load_split({bad!r}) raises DatasetError")

try:
    D.load_mage("train")
    check(False, "load_mage('train') must raise - train.csv is not fetched")
except D.DatasetError:
    check(True, "load_mage('train') raises (train.csv deliberately unfetched)")


# --- loaders ---------------------------------------------------------------

print("\n[2] MAGE splits load with our label convention")

mage_frames = {}
for split in ("test", "valid", "ood_gpt", "ood_gpt_para"):
    df = D.load_mage(split)
    mage_frames[split] = df
    check(list(df.columns) == ["text", "label", "group", "src"],
          f"mage:{split} exact columns")
    check(set(df["label"].unique()) == {0, 1},
          f"mage:{split} has both classes")
    check(int(df["label"].isna().sum()) == 0, f"mage:{split} no null labels")
    check(df["group"].nunique() >= 5,
          f"mage:{split} >=5 groups (bootstrap needs them)")
    # THE invariant: our 0 must line up with '*_human' sources.
    human = df["src"].str.endswith("_human")
    check(bool((df.loc[human, "label"] == 0).all()),
          f"mage:{split} '*_human' sources are label 0 (authentic)")
    check(bool((df.loc[~human, "label"] == 1).all()),
          f"mage:{split} all other sources are label 1 (synthetic)")
    check(human.sum() > 0 and (~human).sum() > 0,
          f"mage:{split} both source kinds present (guard is actually exercised)")

test_df = mage_frames["test"]
check(len(test_df) > 50000, "mage:test is the full testbed (>50k rows)")
check(mage_frames["valid"].shape[0] > 50000, "mage:valid is a full calibration split")
check(len(mage_frames["ood_gpt"]) > 1000, "mage:ood_gpt non-trivial")
check(len(mage_frames["ood_gpt_para"]) > 1000, "mage:ood_gpt_para non-trivial")

# a paraphrased human text is synthetic under our convention
para = mage_frames["ood_gpt_para"]
hp = para[para["src"].str.endswith("_human_para")]
check(len(hp) > 0 and bool((hp["label"] == 1).all()),
      "mage '*_human_para' counts as synthetic (paraphrase = machine edit)")

# Groups are ORIGIN DOMAINS (e.g. 'cmv_human'), which legitimately appear in
# both splits - that is expected. What must not leak between calibration and
# test is the *text*.
vtexts = set(mage_frames["valid"]["text"])
ttexts = set(test_df["text"])
shared = len(vtexts & ttexts)
print(f"    (note: {shared} texts are duplicated between valid and test)")

filtered, removed = D.drop_seen(test_df, vtexts)
check(removed == shared, "drop_seen removes exactly the shared texts")
check(len(set(filtered["text"]) & vtexts) == 0,
      "after drop_seen, zero calibration text leaks into test")
check(len(filtered) > 50000,
      "dedup leaves the testbed effectively intact")
check(len(set(mage_frames["valid"]["group"]) & set(test_df["group"])) > 0,
      "groups overlap across splits (expected: they are origin domains)")


print("\n[3] HC3")

hc3 = D.load_hc3()
check(list(hc3.columns) == ["text", "label", "group", "src"],
      "hc3 exact columns")
check(set(hc3["label"].unique()) == {0, 1}, "hc3 has both classes")
check(len(hc3) > 10000, "hc3 has >10k answer rows")
check(hc3["group"].nunique() >= 5, "hc3 >=5 groups")
check(hc3["src"].nunique() >= 5, "hc3 spans >=5 domains")
check(bool((hc3.loc[hc3["label"] == 0, "text"].str.len() > 0).all()),
      "hc3 authentic rows are non-empty")

# every question that has both kinds must yield both labels in one group
by_group = hc3.groupby("group")["label"].nunique()
check(int((by_group > 1).sum()) > 100,
      "hc3 groups contain both labels (grouping keeps paired answers together)")

sub = D.load_hc3(max_per_source=30)
check(len(sub) < len(hc3), "max_per_source actually truncates")
check(set(sub["label"].unique()) == {0, 1}, "truncated hc3 keeps both classes")


print("\n[4] report")

for name in D.available_splits():
    try:
        print("   ", D.describe(name))
        check(True, f"describe({name})")
    except Exception as exc:
        check(False, f"describe({name}) raised {type(exc).__name__}: {exc}")


print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
