"""Resumable, parallel HTTP-range download of a large file.

The file is cut into fixed-size chunks. Each finished chunk is recorded in a
small sidecar file (<name>.chunks), so an interrupted download resumes where it
stopped. The final size is checked against the server's Content-Length.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CHUNK = 2 * 1024 * 1024
DEADLINE = 45  # seconds allowed per chunk attempt; slower connections are abandoned and retried


def _fetch(url, path, idx, size, done_path, fd):
    start = idx * CHUNK
    end = min(start + CHUNK, size) - 1
    err = None
    for attempt in range(20):
        try:
            t0 = time.time()
            buf = bytearray()
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=15, stream=True) as r:
                if r.status_code != 206:
                    raise IOError(f"bad status {r.status_code}")
                for block in r.iter_content(65536):
                    buf += block
                    if time.time() - t0 > DEADLINE:
                        raise TimeoutError("chunk too slow, retrying on a fresh connection")
            if len(buf) != end - start + 1:
                raise IOError(f"short read {len(buf)}")
            os.pwrite(fd, bytes(buf), start)
            with open(done_path, "a") as f:
                f.write(f"{idx}\n")
            return idx
        except Exception as e:  # retry transient network errors
            err = e
    raise RuntimeError(f"chunk {idx} failed after retries: {err}")


def download(url, path, workers=8):
    size = int(requests.head(url, allow_redirects=True, timeout=30).headers["Content-Length"])
    n = (size + CHUNK - 1) // CHUNK
    done_path = path + ".chunks"
    done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        if os.fstat(fd).st_size != size:
            os.ftruncate(fd, size)
        todo = [i for i in range(n) if i not in done]
        print(f"{n} chunks, {len(done)} already done, {len(todo)} to fetch", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for k, _ in enumerate(ex.map(lambda i: _fetch(url, path, i, size, done_path, fd), todo), 1):
                if k % 20 == 0 or k == len(todo):
                    print(f"  {len(done) + k}/{n} chunks ({(len(done) + k) / n:.0%})", flush=True)
    finally:
        os.close(fd)
    assert os.path.getsize(path) == size, "size mismatch"
    os.remove(done_path)
    print("download complete:", path, size, "bytes")


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
