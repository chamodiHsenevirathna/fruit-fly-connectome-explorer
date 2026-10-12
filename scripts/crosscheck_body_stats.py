"""Cross-check per-body sums from our checkpoint against the dataset's own body-stats table.

Reads only a few record batches of body-stats by HTTP range request (a few MB), joins them to the
per-body in/out tables saved by validate_connections.py (no rescan of the connection table).
"""
import pickle, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc
from connectome.remote import HttpRangeFile

URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-stats-male-cns-v1.0-minconf-0.5.feather"
ck = pickle.load(open("data/interim/validation/checkpoint.pkl", "rb"))
oi, os_, od = ck["out"]; ii, is_, idg = ck["inn"]
rd = ipc.open_file(pa.PythonFile(HttpRangeFile(URL), mode="r"))
n = rd.num_record_batches
ann = pd.read_feather("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=["bodyId", "superclass", "status", "type"])
ann_ids = np.sort(ann.bodyId.to_numpy())

def lookup(ids, tab_ids, tab_vals):
    pos = np.searchsorted(tab_ids, ids); pos[pos == len(tab_ids)] = 0
    hit = tab_ids[pos] == ids
    return np.where(hit, tab_vals[pos], 0)

frames = []
for i in [0, 1, 50, 300, 700, 1000, 1200, n - 1]:
    b = rd.get_batch(i).to_pandas()
    b["batch"] = i
    b["our_in_strength"] = lookup(b.body.to_numpy(), ii, is_); b["our_out_strength"] = lookup(b.body.to_numpy(), oi, os_)
    b["our_in_degree"] = lookup(b.body.to_numpy(), ii, idg); b["our_out_degree"] = lookup(b.body.to_numpy(), oi, od)
    b["in_annotation_file"] = np.isin(b.body.to_numpy(), ann_ids)
    frames.append(b)
df = pd.concat(frames, ignore_index=True)
print("rows sampled:", len(df), "| columns:", list(df.columns[:11]))
for a, b_ in [("post", "our_in_strength"), ("downstream", "our_out_strength"), ("pre", "our_out_degree"), ("synweight", "our_in_strength")]:
    eq = (df[a].astype("int64") == df[b_]).mean()
    print(f"body-stats.{a} == {b_}: {eq:.4f} of sampled bodies")
print("\nper batch: share in annotation file, status_fine counts (top 5), superclass non-null share")
for i, g in df.groupby("batch"):
    print(i, f"n={len(g)} in_annot={g.in_annotation_file.mean():.3f} superclass_notna={g.superclass.notna().mean():.3f}",
          g.status_fine.astype(object).fillna("NA").value_counts().head(4).to_dict())
last = df[df.batch == n - 1]; print("\nlast batch sample:\n", last[["body", "pre", "post", "downstream", "synweight", "our_in_strength", "our_out_strength"]].head(5).to_string())
print("\nsample rows from batch 0 (largest):\n", df[df.batch == 0][["body", "pre", "post", "downstream", "synweight", "rank", "our_in_strength", "our_out_strength", "superclass", "type"]].head(4).to_string())

# Namespace test: for bodies present in BOTH tables, do the labels agree? (body-stats carries its own copy of superclass/type)
j = df[df.in_annotation_file].merge(ann.rename(columns={"bodyId": "body", "superclass": "ann_superclass", "type": "ann_type"})[["body", "ann_superclass", "ann_type"]], on="body")
print("\nbodies in both tables:", len(j))
print("superclass agrees (incl. both null): %.4f" % ((j.superclass.fillna("") == j.ann_superclass.fillna("")).mean()))
print("type agrees (incl. both null):       %.4f" % ((j.type.fillna("") == j.ann_type.fillna("")).mean()))
