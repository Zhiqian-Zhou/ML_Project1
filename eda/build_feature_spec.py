"""Generate eda/feature_spec.csv from eda/column_decisions.csv (normalized feature spec).

The EDA table column_decisions.csv describes each raw column with a free-text
recommended_action. This script maps those actions to a fixed vocabulary, applies
the corrections of the second EDA pass (OVERRIDES), adds the derived features,
validates the result and writes feature_spec.csv. preprocessing.py reads only
feature_spec.csv at run time; this script makes the spec reproducible.

Usage (from any folder):
    python build_feature_spec.py [EDA_DIR] [OUT_CSV] [RAW_PATH]

    EDA_DIR: folder with column_decisions.csv and special_codes.json
        (default: the folder of this script).
    OUT_CSV: output path (default: EDA_DIR/feature_spec.csv).
    RAW_PATH: raw training matrix used to check that every positive value is
        observed, a .npz with key "x_train"
        (default: EDA_DIR/../data/cache/raw.npz; the check is skipped if missing).

Columns of feature_spec.csv: name, source, kind, positive_values, notes. One row
per raw column (raw order), then the derived rows.
"""

import csv
import json
import os
import sys

import numpy as np

KINDS = ("drop", "binary", "ordinal", "onehot", "continuous", "continuous_log")
SPEC_COLUMNS = (
    "name",
    "source",
    "kind",
    "positive_values",
    "notes",
)
FORBIDDEN_TYPES = ("admin", "id", "date", "weight")

# Legacy recommended_action (column_decisions.csv) -> kind. The first matching
# keyword wins, so "drop" is checked before the encoding verbs.
LEGACY_KEYWORDS = (
    ("drop", "drop"),
    ("binary_recode", "binary"),
    ("one_hot", "onehot"),
    ("keep_ordinal", "ordinal"),
    ("keep_continuous", "continuous"),
    ("unit_convert", "continuous"),
)

# Positive raw value(s) of every binary feature, checked against the codebook
# (codebook.txt / calc.txt) and the observed raw values.
POSITIVE_VALUES = {
    "HLTHPLN1": "1",  # 1 = yes, has health coverage
    "MEDCOST": "1",  # 1 = yes, could not see a doctor because of cost
    "BPMEDS": "1",  # 1 = yes, takes blood pressure medicine
    "BLOODCHO": "1",  # 1 = yes, cholesterol ever checked
    "TOLDHI2": "1",  # 1 = yes, told high cholesterol
    "CVDSTRK3": "1",  # 1 = yes, ever had a stroke
    "CHCSCNCR": "1",  # 1 = yes, skin cancer
    "CHCOCNCR": "1",  # 1 = yes, other cancer
    "CHCCOPD1": "1",  # 1 = yes, COPD
    "HAVARTH3": "1",  # 1 = yes, arthritis
    "ADDEPEV2": "1",  # 1 = yes, depressive disorder
    "CHCKIDNY": "1",  # 1 = yes, kidney disease
    "SEX": "1",  # 1 = male, 2 = female
    "VETERAN3": "1",  # 1 = yes, veteran
    "INTERNET": "1",  # 1 = yes, used the internet
    "QLACTLM2": "1",  # 1 = yes, activity limitation
    "USEEQUIP": "1",  # 1 = yes, uses special equipment
    "BLIND": "1",  # 1 = yes, blind or difficulty seeing
    "DECIDE": "1",  # 1 = yes, difficulty concentrating
    "DIFFWALK": "1",  # 1 = yes, difficulty walking
    "DIFFDRES": "1",  # 1 = yes, difficulty dressing
    "DIFFALON": "1",  # 1 = yes, difficulty doing errands alone
    "USENOW3": "1|2",  # 1 = every day, 2 = some days, 3 = not at all
    "EXERANY2": "1",  # 1 = yes, any exercise
    "FLUSHOT6": "1",  # 1 = yes, flu shot
    "PNEUVAC3": "1",  # 1 = yes, pneumonia shot
    "HIVTST6": "1",  # 1 = yes, HIV test
    "INSULIN": "1",  # 1 = yes, takes insulin
    "CIMEMLOS": "1",  # 1 = yes, memory loss
    "WTCHSALT": "1",  # 1 = yes, watching sodium
    "DRADVISE": "1",  # 1 = yes, advised to reduce sodium
    "_RFDRHV5": "2",  # calculated: 1 = no, 2 = yes heavy drinker
    "_PASTRNG": "1",  # calculated: 1 = meets the strength guideline
    "_LMTACT1": "1",  # 1 = arthritis and limited usual activities
    "_LMTWRK1": "1",  # 1 = arthritis and limited work
}

HEAVY_TAIL = "heavy-tailed count: log1p"
LEAKY = "leaky: answered only after a heart attack or a stroke (notebook section 7)"

# Corrections applied on top of the legacy mapping, from the second pass of the
# EDA (notebooks/01_eda.ipynb), plus the leaky rows and the log1p counts.
OVERRIDES = {
    "_STATE": {"kind": "drop", "why": "53 states; one-hot was noise in CV"},
    "QSTVER": {"kind": "drop", "why": "landline/cell split explained by age"},
    "MSCODE": {"kind": "drop", "why": "NaN == cell sample; phone block dropped"},
    "NUMADULT": {"kind": "drop", "why": "source of hh_adults"},
    "HHADULT": {"kind": "drop", "why": "source of hh_adults"},
    "DIABAGE2": {"kind": "drop", "why": "source of diabetes_years"},
    "SMOKE100": {"kind": "drop", "why": "exactly encoded by _SMOKER3"},
    "_RFBING5": {"kind": "drop", "why": "_RFBING5==2 iff DRNK3GE5>=1"},
    "_FRUTSUM": {"kind": "drop", "why": "exact sum FTJUDA1_+FRUTDA1_"},
    "_VEGESUM": {"kind": "drop", "why": "exact sum of the 4 vegetable columns"},
    "_SMOKER3": {"kind": "onehot", "why": "positive rate not monotone"},
    "_LMTACT1": {"kind": "binary", "why": "level 3 == HAVARTH3=2: limited flag"},
    "_LMTWRK1": {"kind": "binary", "why": "level 3 == HAVARTH3=2: limited flag"},
    "_LMTSCL1": {"kind": "ordinal", "why": "positive rate monotone in level"},
    "CHECKUP1": {"kind": "onehot", "why": "positive rate not monotone"},
    "_PACAT1": {"kind": "onehot", "why": "positive rate U-shaped"},
    "RENTHOM1": {"kind": "onehot", "why": "weak but cheap (3 levels)"},
    "USENOW3": {"kind": "binary", "why": "every day or some days vs not at all"},
    "_RFDRHV5": {"kind": "binary", "why": "calculated 1=no 2=yes: positive 2"},
    "_PASTRNG": {"kind": "binary", "why": "1 = meets strength guideline"},
    "TOLDHI2": {"kind": "binary", "why": "NaN == BLOODCHO!=1"},
    "DRADVISE": {"kind": "binary", "why": "NaN == WTCHSALT NaN"},
    "HTIN4": {"kind": "continuous", "why": "NaN inside _BMI5 NaN"},
    "WTKG3": {"kind": "continuous", "why": "NaN inside _BMI5 NaN"},
    "JOINPAIN": {"kind": "continuous", "why": "0-10 pain scale"},
    "PHYSHLTH": {"kind": "continuous", "why": "bounded 0-30 days"},
    "MENTHLTH": {"kind": "continuous", "why": "bounded 0-30 days"},
    "POORHLTH": {"kind": "continuous", "why": "bounded 0-30 days"},
    "PA1MIN_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "PA1VIGM_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "_DRNKWEK": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "DROCDY3_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "AVEDRNK2": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "MAXDRNKS": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "DRNK3GE5": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "CHILDREN": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "STRFREQ_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "FTJUDA1_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "FRUTDA1_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "BEANDAY_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "GRENDAY_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "ORNGDAY_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "VEGEDA1_": {"kind": "continuous_log", "why": HEAVY_TAIL},
    "HAREHAB1": {"kind": "drop", "why": LEAKY},
    "CVDASPRN": {"kind": "drop", "why": LEAKY},
    "RDUCHART": {"kind": "drop", "why": LEAKY},
    "RDUCSTRK": {"kind": "drop", "why": LEAKY},
    "RLIVPAIN": {"kind": "drop", "why": LEAKY},
    "ASPUNSAF": {"kind": "drop", "why": LEAKY},
    "STREHAB1": {"kind": "drop", "why": LEAKY},
}

# Derived features appended after the raw rows (computed by
# preprocessing.add_derived; "source" documents the recipe and its raw inputs).
DERIVED_ROWS = (
    {
        "name": "hh_adults",
        "source": "derive:coalesce(NUMADULT;HHADULT)",
        "kind": "continuous_log",
        "positive_values": "",
        "notes": "adults in household: landline NUMADULT else cell HHADULT",
    },
    {
        "name": "diabetes_years",
        "source": "derive:diabetes_years(_AGE80;DIABAGE2;DIABETE3)",
        "kind": "continuous",
        "positive_values": "",
        "notes": "max(_AGE80-DIABAGE2;0) if diabetic; 0 if DIABETE3 in 2|3|4",
    },
)


def legacy_kind(action):
    """Map a free-text recommended_action of column_decisions.csv to a kind.

    Args:
        action: str, legacy action (e.g. "binary_recode + missing_indicator").

    Returns:
        str, one of KINDS.
    """
    for keyword, kind in LEGACY_KEYWORDS:
        if keyword in action:
            return kind
    raise ValueError("unknown legacy action: %r" % action)


def build_rows(decisions):
    """Build the spec rows from the legacy decisions, OVERRIDES and DERIVED_ROWS.

    Args:
        decisions: list of dict, rows of column_decisions.csv (raw order).

    Returns:
        rows: list of dict with the SPEC_COLUMNS keys.
        diffs: list of (name, legacy_kind, new_kind, reason) for every raw row
            whose kind differs from the legacy action.
    """
    rows, diffs = [], []
    for dec in decisions:
        name = dec["name"]
        action = dec["recommended_action"]
        old_kind = legacy_kind(action)
        row = {
            "name": name,
            "source": "",
            "kind": old_kind,
            "positive_values": "",
            "notes": dec["reason"],
        }
        over = OVERRIDES.get(name)
        if over is not None:
            row["kind"] = over["kind"]
            row["notes"] = over["why"]
            if row["kind"] != old_kind:
                diffs.append((name, old_kind, row["kind"], over["why"]))
        if row["kind"] == "binary":
            row["positive_values"] = POSITIVE_VALUES.get(name, "")
        rows.append(row)
    for der in DERIVED_ROWS:
        rows.append(dict(der))
    return rows, diffs


def validate_rows(rows, decisions, codes, x_raw=None):
    """Check the spec rows (kinds, binary positive values, special codes).

    Args:
        rows: list of dict, output of build_rows.
        decisions: list of dict, rows of column_decisions.csv.
        codes: dict, parsed special_codes.json.
        x_raw: optional numpy array (N, n_raw) of raw training values in the raw
            column order, used to check that each positive value is observed.

    Returns:
        List of str, one message per problem (empty when the spec is valid).
    """
    errors = []
    raw_names = [d["name"] for d in decisions]
    types = {d["name"]: d["type"] for d in decisions}
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        errors.append("duplicate names")
    if names[: len(raw_names)] != raw_names:
        errors.append("raw rows are not in raw order")
    for j, row in enumerate(rows):
        name, kind = row["name"], row["kind"]
        if kind not in KINDS:
            errors.append("%s: unknown kind %r" % (name, kind))
        if (kind == "binary") != bool(row["positive_values"]):
            errors.append("%s: positive_values iff kind=binary" % name)
        if row["source"]:
            if not row["source"].startswith("derive:"):
                errors.append("%s: bad source %r" % (name, row["source"]))
            args = row["source"].split("(", 1)[1].rstrip(")").split(";")
            for arg in args:
                if arg not in raw_names:
                    errors.append("%s: unknown source column %s" % (name, arg))
            continue
        if kind == "drop":
            continue
        if types.get(name) in FORBIDDEN_TYPES:
            errors.append("%s: non-drop row of type %s" % (name, types[name]))
        if name not in codes:
            errors.append("%s: missing from special_codes.json" % name)
            continue
        entry = codes[name]
        if kind != "binary":
            continue
        changed = set(entry.get("nan_codes", [])) | set(entry.get("zero_codes", []))
        changed |= set(int(k) for k in entry.get("recode", {}))
        if "scale" in entry or "transform=" in entry.get("notes", ""):
            errors.append("%s: binary column with scale or unit transform" % name)
        values = row["positive_values"].split("|") if row["positive_values"] else []
        for val in values:
            if float(val) in changed:
                errors.append("%s: positive value %s is a special code" % (name, val))
            if x_raw is not None and not np.any(x_raw[:, j] == float(val)):
                errors.append("%s: positive value %s never observed" % (name, val))
    return errors


def write_spec(rows, path):
    """Write the spec rows to a csv file (deterministic, "\\n" line endings).

    Args:
        rows: list of dict with the SPEC_COLUMNS keys.
        path: str, output csv path.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(SPEC_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in SPEC_COLUMNS])


def load_raw(path):
    """Load the raw training matrix for the positive-value check.

    Args:
        path: str, .npz file with key "x_train".

    Returns:
        numpy array (N, 321), or None if the file does not exist.
    """
    if not path or not os.path.exists(path):
        return None
    with np.load(path) as data:
        return data["x_train"]


def main(argv):
    """Build, validate and write the feature spec, then print the diff report.

    Args:
        argv: list of str, optional [EDA_DIR, OUT_CSV, RAW_PATH].

    Returns:
        int, 0 on success, 1 if the validation failed (nothing is written).
    """
    eda_dir = argv[0] if len(argv) > 0 else os.path.dirname(os.path.abspath(__file__))
    out_path = argv[1] if len(argv) > 1 else os.path.join(eda_dir, "feature_spec.csv")
    default_raw = os.path.join(eda_dir, os.pardir, "data", "cache", "raw.npz")
    raw_path = argv[2] if len(argv) > 2 else default_raw
    with open(os.path.join(eda_dir, "column_decisions.csv"), newline="") as f:
        decisions = list(csv.DictReader(f))
    with open(os.path.join(eda_dir, "special_codes.json")) as f:
        codes = json.load(f)
    rows, diffs = build_rows(decisions)
    x_raw = load_raw(raw_path)
    if x_raw is None:
        print("raw data not found (%s): positive-value check skipped" % raw_path)
    elif x_raw.shape[1] != len(decisions):
        print("raw data has %d columns, expected %d" % (x_raw.shape[1], len(decisions)))
        return 1
    errors = validate_rows(rows, decisions, codes, x_raw)
    if errors:
        print("feature spec INVALID:")
        for msg in errors:
            print("  " + msg)
        return 1
    write_spec(rows, out_path)
    print("Kind changes vs legacy recommended_action (%d rows):" % len(diffs))
    for name, old, new, why in diffs:
        print("  %-9s %-11s -> %-14s %s" % (name, old, new, why))
    kept = [r for r in rows if r["kind"] != "drop"]
    count = {}
    for r in kept:
        count[r["kind"]] = count.get(r["kind"], 0) + 1
    print("non-drop rows: %d %s" % (len(kept), sorted(count.items())))
    print("wrote %s (%d rows)" % (out_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
