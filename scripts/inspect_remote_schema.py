"""Print schema + batch layout of a remote Feather file using only HTTP range requests."""
import sys
from connectome.remote import inspect_feather

(schema, nb, meta, stats), reader = inspect_feather(sys.argv[1])
print(schema)
print("record batches:", nb)
rows = [reader.get_batch(i).num_rows for i in range(nb)] if len(sys.argv) > 2 and sys.argv[2] == "--rows" else None
if rows: print("rows per batch (first 5):", rows[:5], "total:", sum(rows))
print("schema metadata:", {k.decode(): v.decode()[:500] for k, v in (meta or {}).items()})
print("requests, bytes fetched, file size:", stats)
