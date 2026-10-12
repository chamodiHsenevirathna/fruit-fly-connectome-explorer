"""Read Arrow Feather (IPC file) metadata over HTTP without downloading the whole file.

An Arrow IPC file stores its schema and the location of every record batch in a
small "footer" at the END of the file. HTTP range requests let us fetch just
those few bytes, so we can learn the columns and row counts of a 1 GB file
while transferring only kilobytes.
"""
import io

import pyarrow as pa
import pyarrow.ipc as ipc
import requests


class HttpRangeFile(io.RawIOBase):
    """Minimal read-only, seekable file object backed by HTTP Range requests."""

    def __init__(self, url, session=None):
        self.url = url
        self.s = session or requests.Session()
        head = self.s.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges") != "bytes":
            raise RuntimeError("Server does not advertise byte-range support")
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0
        self.requests_made = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=60)
        if r.status_code != 206 or len(r.content) != n:
            raise RuntimeError(f"Range request failed: status {r.status_code}, got {len(r.content)} of {n} bytes")
        self.pos += n
        self.bytes_fetched += n
        self.requests_made += 1
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def inspect_feather(url):
    """Return (schema, num_record_batches, custom_metadata, fetch_stats) using only the file footer."""
    f = HttpRangeFile(url)
    reader = ipc.open_file(pa.PythonFile(f, mode="r"))
    out = (reader.schema, reader.num_record_batches, reader.schema.metadata, (f.requests_made, f.bytes_fetched, f.size))
    return out, reader
