"""Streaming per-body aggregation of the connection table at several weight thresholds.

Why this design (plain English)
-------------------------------
The connection table has 152 million rows and 88 million distinct bodies, but only ~212 thousand bodies have an
annotation row. We keep one row of numbers per ANNOTATED body (a small dense table) and only running totals for the
rest. Each record batch is folded into those tables and then discarded, so memory stays small and no graph object
is ever built.

Definitions (fixed here, used everywhere)
-----------------------------------------
record      one row (body_pre, body_post, weight). Verified: every directed pair occurs exactly once, so a record
            IS a distinct directed pair and out-degree = number of records (we do NOT deduplicate; see docs).
threshold T keep a record if weight >= T.
population  a set of annotated bodies treated as "neurons" (primary / sensitivity). It is FIXED across thresholds:
            a neuron with no surviving record simply has zeros.
partner     for each body, the other end of a record is either "n" (a member of the population) or "o" (anything
            else: an annotated non-member or an unannotated body).

Per annotated body and threshold we store 8 numbers:
    out_deg_n / out_str_n   distinct downstream partners / summed weight, partner in the population
    out_deg_o / out_str_o   same, partner outside the population
    in_deg_n  / in_str_n    distinct upstream partners / summed weight, partner in the population
    in_deg_o  / in_str_o    same, partner outside the population
For a NEURON, *_n is neuron-to-neuron connectivity (used for the primary ranking) and *_o is its connectivity to
other bodies. Edge categories (source type -> target type): NN, NO, ON, OO are also totalled per threshold.
"""
import numpy as np

METRICS = ["out_deg_n", "out_str_n", "out_deg_o", "out_str_o", "in_deg_n", "in_str_n", "in_deg_o", "in_str_o"]
CATEGORIES = ["NN", "NO", "ON", "OO"]          # source -> target; N = population member, O = any other body
GLOBAL_COLS = ["records", "strength", "src_unann_records", "src_unann_strength", "dst_unann_records", "dst_unann_strength"]
M = {name: i for i, name in enumerate(METRICS)}


class Aggregator:
    def __init__(self, ann_ids, populations, thresholds):
        self.ann_ids = np.asarray(ann_ids, dtype=np.int64)
        if len(self.ann_ids) > 1 and not np.all(np.diff(self.ann_ids) > 0):
            raise ValueError("ann_ids must be strictly increasing (sorted, unique)")
        self.n = len(self.ann_ids)
        self.thresholds = list(thresholds)
        # population masks get one extra trailing False so that index -1 (= unannotated body) means "not a member"
        self.pop_names = list(populations)
        self._pm = {p: np.append(np.asarray(m, dtype=bool), False) for p, m in populations.items()}
        nt = len(self.thresholds)
        self.per_body = {p: np.zeros((nt, len(METRICS), self.n), dtype=np.int64) for p in self.pop_names}
        self.cat = {p: np.zeros((nt, len(CATEGORIES), 2), dtype=np.int64) for p in self.pop_names}   # [..., 0]=records [..., 1]=strength
        self.glob = np.zeros((nt, len(GLOBAL_COLS)), dtype=np.int64)
        self.rows_seen = 0

    def _index(self, ids):
        pos = np.searchsorted(self.ann_ids, ids)
        pos[pos == self.n] = 0
        return np.where(self.ann_ids[pos] == ids, pos, -1)

    def _bin(self, idx, w=None):
        b = np.bincount(idx, weights=w, minlength=self.n)
        return b if w is None else np.rint(b)

    def update(self, pre, post, w):
        """Fold one batch of records into the running tables."""
        pre = np.asarray(pre, dtype=np.int64); post = np.asarray(post, dtype=np.int64); w = np.asarray(w, dtype=np.int64)
        ip, iq = self._index(pre), self._index(post)
        self.rows_seen += len(pre)
        for t, T in enumerate(self.thresholds):
            keep = w >= T
            if not keep.any():
                continue
            ipk, iqk, wk = ip[keep], iq[keep], w[keep]
            wf = wk.astype(np.float64)
            self.glob[t] += [len(wk), wk.sum(), (ipk < 0).sum(), wk[ipk < 0].sum(), (iqk < 0).sum(), wk[iqk < 0].sum()]
            for p in self.pop_names:
                pm = self._pm[p]
                src_n, dst_n = pm[ipk], pm[iqk]          # index -1 -> last element -> False
                for c, (s, d) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
                    sel = (src_n == s) & (dst_n == d)
                    self.cat[p][t, c] += [sel.sum(), wk[sel].sum()]
                acc = self.per_body[p][t]
                a_src, a_dst = ipk >= 0, iqk >= 0
                for tag, cond in (("n", dst_n), ("o", ~dst_n)):          # outgoing, by partner type
                    sel = a_src & cond
                    acc[M[f"out_deg_{tag}"]] += self._bin(ipk[sel]).astype(np.int64)
                    acc[M[f"out_str_{tag}"]] += self._bin(ipk[sel], wf[sel]).astype(np.int64)
                for tag, cond in (("n", src_n), ("o", ~src_n)):          # incoming, by partner type
                    sel = a_dst & cond
                    acc[M[f"in_deg_{tag}"]] += self._bin(iqk[sel]).astype(np.int64)
                    acc[M[f"in_str_{tag}"]] += self._bin(iqk[sel], wf[sel]).astype(np.int64)

    # ---- consistency checks -------------------------------------------------------------------------------
    def check_identities(self):
        """Return a list of (description, ok) for the sum-of-incoming == sum-of-outgoing identities."""
        out = []
        for p in self.pop_names:
            member = self._pm[p][:-1]
            for t, T in enumerate(self.thresholds):
                pb, cat = self.per_body[p][t], self.cat[p][t]
                nn_out = pb[M["out_str_n"]][member].sum(); nn_in = pb[M["in_str_n"]][member].sum()
                out.append((f"{p} T>={T}: NN out==in==category total", nn_out == nn_in == cat[0, 1]))
                out.append((f"{p} T>={T}: NO category == members' out_str_o", pb[M["out_str_o"]][member].sum() == cat[1, 1]))
                out.append((f"{p} T>={T}: ON category == members' in_str_o", pb[M["in_str_o"]][member].sum() == cat[2, 1]))
                out.append((f"{p} T>={T}: categories sum to all retained records/weight",
                            cat[:, 0].sum() == self.glob[t, 0] and cat[:, 1].sum() == self.glob[t, 1]))
                # records with an annotated source are all counted in some body's out table (and likewise for targets)
                all_out = pb[M["out_str_n"]].sum() + pb[M["out_str_o"]].sum() + self.glob[t, 3]
                all_in = pb[M["in_str_n"]].sum() + pb[M["in_str_o"]].sum() + self.glob[t, 5]
                out.append((f"{p} T>={T}: all out-strength (+unannotated sources) == all in-strength (+unannotated targets) == total",
                            all_out == all_in == self.glob[t, 1]))
        return out

    # ---- persistence ---------------------------------------------------------------------------------------
    def state(self):
        d = {"rows_seen": np.array(self.rows_seen), "glob": self.glob, "thresholds": np.array(self.thresholds), "ann_ids": self.ann_ids}
        for p in self.pop_names:
            d[f"per_body_{p}"] = self.per_body[p]; d[f"cat_{p}"] = self.cat[p]
        return d

    def load_state(self, d):
        if list(d["thresholds"]) != self.thresholds or not np.array_equal(d["ann_ids"], self.ann_ids):
            raise ValueError("Checkpoint does not match the current thresholds/annotation table")
        self.rows_seen = int(d["rows_seen"]); self.glob = d["glob"].copy()
        for p in self.pop_names:
            self.per_body[p] = d[f"per_body_{p}"].copy(); self.cat[p] = d[f"cat_{p}"].copy()
