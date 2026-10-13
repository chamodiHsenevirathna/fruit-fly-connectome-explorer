"""Population filters: the provisional neuron definitions (see annotations.py)."""
import numpy as np
import pandas as pd
import pytest

from connectome.annotations import PRIMARY, SENSITIVITY, load_annotations, make_populations


def _write(tmp_path, df):
    p = tmp_path / "ann.feather"; df.to_feather(p); return p


def _table():
    # bodyId: 3 has superclass + Traced; 1 has superclass but Orphan; 2 has no superclass but is Traced; 4 has neither; 5 has superclass + Traced
    return pd.DataFrame({"bodyId": [5, 1, 3, 2, 4], "superclass": ["x", "y", "z", None, None],
                         "status": pd.Categorical(["Traced", "Orphan", "Traced", "Traced", None]), "type": list("abcde")})


def test_populations_follow_the_rules_and_rows_are_sorted_by_body(tmp_path):
    ann = load_annotations(_write(tmp_path, _table()))
    assert ann.bodyId.tolist() == [1, 2, 3, 4, 5]
    pops = make_populations(ann)
    assert ann.bodyId[pops[PRIMARY]].tolist() == [1, 3, 5]            # superclass not null
    assert ann.bodyId[pops[SENSITIVITY]].tolist() == [3, 5]           # superclass not null AND status == "Traced"
    assert set(ann.bodyId[pops[SENSITIVITY]]) <= set(ann.bodyId[pops[PRIMARY]])   # sensitivity is a subset of primary


def test_status_other_than_traced_and_missing_status_excluded_from_sensitivity(tmp_path):
    pops = make_populations(load_annotations(_write(tmp_path, _table())))
    assert pops[SENSITIVITY].sum() == 2                                # Orphan (body 1) and missing status are not Traced


def test_missing_required_column_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        load_annotations(_write(tmp_path, _table().drop(columns=["superclass"])))


def test_duplicate_or_null_body_ids_rejected(tmp_path):
    dup = _table(); dup.loc[1, "bodyId"] = 5
    with pytest.raises(ValueError, match="duplicated"):
        load_annotations(_write(tmp_path, dup))
    nul = _table().astype({"bodyId": "float64"}); nul.loc[0, "bodyId"] = np.nan
    with pytest.raises(ValueError, match="null"):
        load_annotations(_write(tmp_path, nul))
