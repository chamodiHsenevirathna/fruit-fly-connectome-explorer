"""Fetch a few record batches (a tiny fraction of the file) by range request to examine row semantics."""
import numpy as np, pyarrow as pa, pyarrow.ipc as ipc, collections
from connectome.remote import HttpRangeFile
URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
f = HttpRangeFile(URL); r = ipc.open_file(pa.PythonFile(f, mode="r"))
n = r.num_record_batches
print("batches", n)
for i in [0, 1, n // 4, n // 2, 3 * n // 4, n - 1]:
    t = r.get_batch(i); a = t.column(0).to_numpy(); b = t.column(1).to_numpy(); w = t.column(2).to_numpy()
    key = np.stack([a, b], 1); uniq = np.unique(key, axis=0).shape[0]
    print(f"batch {i}: rows={len(a)} pre[min,max]=({a.min()},{a.max()}) sorted_by_pre={bool(np.all(np.diff(a)>=0))} "
          f"unique_pairs={uniq} weight[min,med,max]=({w.min()},{int(np.median(w))},{w.max()}) null={t.column(2).null_count} self_loops={int((a==b).sum())}")
print("fetched MB:", f.bytes_fetched/1e6, "of", f.size/1e6)
