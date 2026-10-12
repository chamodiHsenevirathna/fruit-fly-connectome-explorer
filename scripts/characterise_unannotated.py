"""Describe bodies in the connection table that have no annotation row, using the saved checkpoint (no rescan)."""
import pickle, numpy as np, pandas as pd
ck = pickle.load(open("data/interim/validation/checkpoint.pkl", "rb"))
oi, os_, od = ck["out"]; ii, is_, idg = ck["inn"]
ann = np.sort(pd.read_feather("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=["bodyId"]).bodyId.to_numpy().copy())
def m(x):
    p = np.searchsorted(ann, x); p[p == len(ann)] = 0; return ann[p] == x
mo, mi = m(oi), m(ii)
print("OUT table (bodies with >=1 outgoing record): total %d, annotated %d, unannotated %d" % (len(oi), mo.sum(), (~mo).sum()))
print("IN  table (bodies with >=1 incoming record): total %d, annotated %d, unannotated %d" % (len(ii), mi.sum(), (~mi).sum()))
u_in, u_out = is_[~mi], os_[~mo]
print("UNANNOTATED, incoming strength: ==1 %.4f  <=2 %.4f  <=5 %.4f  >=10: %d  >=100: %d  max %d" % ((u_in == 1).mean(), (u_in <= 2).mean(), (u_in <= 5).mean(), (u_in >= 10).sum(), (u_in >= 100).sum(), u_in.max()))
print("UNANNOTATED, outgoing strength: ==1 %.4f  >=10: %d  >=100: %d  max %d" % ((u_out == 1).mean(), (u_out >= 10).sum(), (u_out >= 100).sum(), u_out.max()))
tot = int(is_.sum())
print("share of total weight with unannotated POST end %.4f ; unannotated PRE end %.4f" % (is_[~mi].sum() / tot, os_[~mo].sum() / tot))
print("ANNOTATED: median in-strength %d, median out-strength %d; share of total weight in %.4f / out %.4f" % (np.median(is_[mi]), np.median(os_[mo]), is_[mi].sum() / tot, os_[mo].sum() / tot))
