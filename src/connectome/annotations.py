"""Annotation loading and the (provisional) neuron populations.

Both populations are PROVISIONAL: they rest on `superclass` being an annotation given to neurons only,
which the preprint states in prose but the dataset download page does not define. Treat results as conditional on that assumption.
"""
import numpy as np
import pandas as pd

REQUIRED = ["bodyId", "superclass", "status"]
OPTIONAL = ["type", "statusLabel", "somaSide", "class", "subclass", "instance"]

PRIMARY = "primary"          # superclass is not null
SENSITIVITY = "sensitivity"  # superclass is not null AND status == "Traced"; public name: "fully traced subset" (a sensitivity analysis, not a better definition)


def load_annotations(path):
    """Read the annotation table, check required columns, and return it sorted by bodyId."""
    head = pd.read_feather(path)
    missing = [c for c in REQUIRED if c not in head.columns]
    if missing:
        raise ValueError(f"Annotation file is missing required columns {missing}; found {list(head.columns)[:12]}...")
    keep = REQUIRED + [c for c in OPTIONAL if c in head.columns]
    ann = head[keep].copy()
    if ann.bodyId.isna().any():
        raise ValueError("Annotation table has null bodyId values")
    if ann.bodyId.duplicated().any():
        raise ValueError("Annotation table has duplicated bodyId values; one row per body is expected")
    ann["bodyId"] = ann.bodyId.astype("int64")
    return ann.sort_values("bodyId").reset_index(drop=True)


def make_populations(ann):
    """Boolean masks (aligned with `ann` rows) for the primary and sensitivity neuron populations."""
    has_super = ann["superclass"].notna().to_numpy()
    traced = (ann["status"].astype(object) == "Traced").to_numpy()
    return {PRIMARY: has_super, SENSITIVITY: has_super & traced}
