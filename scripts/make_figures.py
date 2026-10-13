"""Build the research figures from the saved result tables (no re-run of the aggregation).

Usage (from the repository root):  PYTHONPATH=src python scripts/make_figures.py [--no-fig7]

Reads  outputs/tables/*.csv  and  data/interim/aggregation/aggregates.npz  (the latter only for the rank-trajectory
table and the optional distribution figure). Writes
  outputs/figures/fig1..fig7 *.png   (300 dpi)
  outputs/figures/README.md          captions, threshold definitions, population definitions, filtering notes, attribution
  outputs/tables/rank_trajectories_<metric>.csv   the reproducible table behind Figure 6
Every number in a figure or caption is computed from those tables; the script first cross-checks the tables against
each other and stops if they disagree.

Colour: categorical colours are the first four slots of a pink/lilac-led palette that passed a colour-palette
validator in light mode on the chart surface (lightness band, chroma floor, adjacent-pair colour-blind separation,
normal-vision floor, contrast). Identity is never colour alone: markers, line styles and direct labels back it up.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from connectome import figdata as fd
from connectome.aggregate import M
from connectome.annotations import load_annotations, make_populations

T_OUT, F_OUT = "outputs/tables", "outputs/figures"
THR = list(fd.THRESHOLDS)
POP, POP_LABEL = "primary", fd.POPULATION_LABELS["primary"]
POPN = 0          # size of the fixed primary population; set from the saved tables in main()
METRICS = {"out_strength_nn": ("Out-strength", "out_str_n", "summed weight to neuron partners"),
           "out_degree_nn": ("Out-degree", "out_deg_n", "number of distinct downstream neuron partners")}

# ---- palette (validated) and ink -------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fbf9fb", "#2b2530", "#5a5160", "#8a8190", "#e8e2ea"
VIOLET, TEAL, OLIVE, ROSE = "#6f55a8", "#0f95a3", "#6d8f2a", "#d4608a"          # validated slots 1-4, in this order
K_RAMP = {10: "#4a3680", 50: "#7b62b8", 100: "#ab98d6"}                          # one hue, light->dark by k (ordinal)
K_MARK = {10: "o", 50: "s", 100: "^"}
CAT_COLOR = {"NN": VIOLET, "NO": TEAL, "ON": OLIVE, "OO": ROSE}
CAT_NAME = {"NN": "neuron → neuron", "NO": "neuron → other body", "ON": "other body → neuron", "OO": "other body → other body"}
SUPER_STYLE = {"ol_intrinsic": (VIOLET, "o"), "cb_intrinsic": (TEAL, "s"), "ascending_neuron": (OLIVE, "^"), "visual_centrifugal": (ROSE, "D")}
CMAP = LinearSegmentedColormap.from_list("lilac", ["#f3eff9", "#b9a8dc", "#6f55a8", "#2f2158"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED, "axes.titlesize": 10,
                     "axes.titleweight": "regular", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
                     "lines.linewidth": 1.6, "lines.markersize": 5.5, "savefig.dpi": 300})
XL = [f"≥{t}" for t in THR]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def save(fig, name):
    os.makedirs(F_OUT, exist_ok=True)
    path = f"{F_OUT}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def place_end_labels(ax, items, x_text, min_gap_pt=9.5, fontsize=7.2, leader=True):
    """Direct labels at the right end of lines, spread so none overlap and kept inside the axes.

    items: list of (x_end, y_end, text, colour). Axes limits/scales must already be set.
    """
    if not items:
        return
    fig = ax.figure; fig.canvas.draw()
    tr = ax.transData; inv = tr.inverted()
    px = np.array([tr.transform((x, y))[1] for x, y, _, _ in items]); gap = min_gap_pt * fig.dpi / 72
    order = np.argsort(px); new = px.copy()
    for lo_i, hi_i in zip(order[:-1], order[1:]):
        if new[hi_i] - new[lo_i] < gap:
            new[hi_i] = new[lo_i] + gap
    new += px.mean() - new.mean()
    bb = ax.get_window_extent(); lo, hi = bb.y0 + gap / 2, bb.y1 - gap / 2
    if new.max() > hi: new -= new.max() - hi
    if new.min() < lo: new += lo - new.min()
    x_px = tr.transform((x_text, 1))[0]
    for (x, y, text, c), ypx in zip(items, new):
        y_txt = inv.transform((x_px, ypx))[1]
        if leader and abs(ypx - tr.transform((x, y))[1]) > 2:
            ax.plot([x + 0.06, x_text - 0.04], [y, y_txt], color=c, lw=0.5, alpha=0.6, clip_on=False, zorder=2)
        ax.text(x_text, y_txt, text, fontsize=fontsize, color=INK, va="center", ha="left", clip_on=False)


def footer(fig, text, y=-0.02):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.5, color=INK2, wrap=True)


# ---------------------------------------------------------------------------------------------------- Figure 1
def fig1(tot_ret, hist):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    h = hist[hist.weight >= 1]
    ax.scatter(h.weight, h.n_records, s=14, color=VIOLET, alpha=0.85, linewidths=0, zorder=3)
    total_records = int(hist.n_records.sum()); share_w1 = float(hist[hist.weight == 1].n_records.sum() / total_records)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, 3500); ax.set_ylim(0.6, 2e9)
    for _, r in tot_ret.iterrows():
        T = int(r.threshold)
        ax.axvline(T, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(T * 1.03, 1.6e9, f"≥{T}\n{pct(r.records_share, 2 if r.records_share < 0.01 else 1)}", fontsize=7.5, color=INK, va="top", ha="left")
    ax.set_xlabel("connection weight w (synaptic connection records between a pair of bodies), log scale")
    ax.set_ylabel("number of body pairs with weight w, log scale")
    ax.set_title(f"Most connections are weak: {pct(share_w1, 0)} of body pairs have weight 1", loc="left")
    ax.text(0.99, 0.62, f"dashed lines: thresholds T (keep records with weight ≥ T)\nnumber under each T: share of all {total_records / 1e6:.1f} M records kept",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    footer(fig, f"All {total_records:,} connection records of MaleCNS v1.0 (one per distinct directed body pair; no filtering). {len(hist)} distinct weight values. "
                "Descriptive only: no claim is made about the functional form of the distribution.")
    save(fig, "fig1_weight_distribution.png")


# ---------------------------------------------------------------------------------------------------- Figure 2
def fig2(tot_ret, shares):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(THR))
    a.plot(x, 100 * tot_ret.records_share, color=VIOLET, marker="o", label="records (body pairs)")
    a.plot(x, 100 * tot_ret.weight_share, color=TEAL, marker="s", label="summed weight (synaptic connection records)")
    a.set_yscale("log"); a.set_ylim(0.08, 150); a.set_yticks([0.1, 1, 10, 100]); a.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    a.set_xticks(x); a.set_xticklabels(XL); a.set_xlim(-0.3, len(THR) - 0.3 + 0.5)
    place_end_labels(a, [(x[-1], 100 * tot_ret[k_].iloc[-1], pct(tot_ret[k_].iloc[-1], 2 if tot_ret[k_].iloc[-1] < 0.01 else 1), c_) for k_, c_ in (("records_share", VIOLET), ("weight_share", TEAL))], x[-1] + 0.3, fontsize=8)
    a.set_xlabel("threshold T (keep records with weight ≥ T)"); a.set_ylabel("share of all records / of all weight kept (log scale)")
    a.set_title("(a) Share of the graph that survives", loc="left"); a.legend(loc="lower left")
    sty = {"any": (VIOLET, "o", "any neuron-to-neuron connection"), "outgoing": (TEAL, "s", "an outgoing connection to a neuron"),
           "incoming": (OLIVE, "^", "an incoming connection from a neuron"), "none": (ROSE, "D", "no neuron-to-neuron connection")}
    for key, (c, m, lab) in sty.items():
        b.plot(x, 100 * shares[key], color=c, marker=m, label=lab)
    n = int(shares.population_size.iloc[0])
    b.set_ylim(0, 100); b.set_yticks([0, 25, 50, 75, 100]); b.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    b.set_xticks(x); b.set_xticklabels(XL); b.set_xlim(-0.3, len(THR) - 0.3 + 0.5)
    place_end_labels(b, [(x[-1], 100 * shares[k_].iloc[-1], pct(shares[k_].iloc[-1], 0), sty[k_][0]) for k_ in sty], x[-1] + 0.3, fontsize=8)
    b.set_xlabel("threshold T (keep records with weight ≥ T)"); b.set_ylabel(f"share of the fixed {n:,} neurons")
    b.set_title(f"(b) Neurons that keep a connection ({POP_LABEL})", loc="left"); b.legend(loc="center left", bbox_to_anchor=(0.0, 0.52))
    footer(fig, f"(a) relative to threshold 1 ({int(tot_ret.records.iloc[0]):,} records; {int(tot_ret.strength.iloc[0]):,} summed weight). (b) The population ({POP_LABEL}: bodies with a superclass annotation) is the same at every threshold; "
                "“none” counts neurons with no neuron-to-neuron connection at all (they may still connect to other bodies).", y=-0.03)
    save(fig, "fig2_retention.png")


# ---------------------------------------------------------------------------------------------------- Figure 3
def fig3(cat):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.22})
    x = np.arange(len(THR))
    for ax, key, title in ((axes[0], "records_share", "(a) by number of records"), (axes[1], "weight_share", "(b) by summed weight")):
        bottom = np.zeros(len(THR))
        for c in fd.CATEGORIES:
            v = np.array([cat[(cat.threshold == T) & (cat.category == c)][key].iloc[0] for T in THR]) * 100
            ax.bar(x, v, bottom=bottom, width=0.72, color=CAT_COLOR[c], edgecolor=SURFACE, linewidth=1.6, label=f"{c}: {CAT_NAME[c]}", zorder=3)
            for xi, vi, bi in zip(x, v, bottom):
                if vi >= 7:
                    ax.text(xi, bi + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels(XL); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
        ax.set_xlabel("threshold T (keep records with weight ≥ T)"); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of retained edges")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.03), fontsize=8)
    footer(fig, f"Edge categories (source → target): N = a neuron of the {POP_LABEL} (bodies with a superclass annotation); O = “other body” = every body that is NOT in that population, i.e. annotated bodies "
                "without a superclass plus all unannotated bodies. Segments are separated by a gap; labels shown for segments of at least 7%.", y=-0.09)
    fig.suptitle("Raising the threshold changes what kind of graph is left, not only how big it is", x=0.01, ha="left", y=1.0, fontsize=10.5)
    save(fig, "fig3_edge_categories.png")


# ---------------------------------------------------------------------------------------------------- Figure 4
def fig4(stabs, tie_info):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True, gridspec_kw={"hspace": 0.42, "wspace": 0.08})
    x = {T: i for i, T in enumerate(THR)}
    for col, (metric, (name, _, desc)) in enumerate(METRICS.items()):
        for row, ref in enumerate((1, 10)):
            ax = axes[row, col]; d = fd.jaccard_vs_reference(stabs[metric], ref)
            ends = []
            for k in (10, 50, 100):
                head = d[(d.k == k) & (d.tie_policy == "id")]; inc = d[(d.k == k) & (d.tie_policy == "inclusive")]
                # the reference list compared with itself has similarity 1 by definition; it is included so each line starts from it
                pts = sorted([(x[ref], 1.0, False)] + [(x[int(r_.threshold)], r_.jaccard, bool(r_.boundary_tie)) for _, r_ in head.iterrows()])
                xs = np.array([p_[0] for p_ in pts]); ys = np.array([p_[1] for p_ in pts]); tie = np.array([p_[2] for p_ in pts])
                ax.plot(xs, ys, color=K_RAMP[k], marker=K_MARK[k], zorder=4)
                if tie.any():
                    ax.scatter(xs[tie], ys[tie], s=62, facecolor=SURFACE, edgecolor=K_RAMP[k], linewidths=1.5, marker=K_MARK[k], zorder=5)
                ax.scatter(np.array([x[int(t)] for t in inc.threshold]) + 0.12, inc.jaccard, marker="+", s=30, color=K_RAMP[k], linewidths=1.0, zorder=5)
                ends.append((xs[-1], ys[-1], f"top {k}", K_RAMP[k]))
            ax.set_xticks(range(len(THR))); ax.set_xticklabels(XL); ax.set_xlim(-0.4, len(THR) - 0.2 + 0.6); ax.set_ylim(-0.03, 1.12)
            ax.annotate("reference", (x[ref], 1.0), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
            place_end_labels(ax, ends, len(THR) - 1 + 0.45, fontsize=7.5)
            ax.set_title(f"{name}: overlap of the top-K list with the T≥{ref} list", loc="left", fontsize=9.5)
            if col == 0: ax.set_ylabel("Jaccard similarity of top-K sets")
            if row == 1: ax.set_xlabel("threshold T being compared")
    handles = [Line2D([], [], color=K_RAMP[k], marker=K_MARK[k], label=f"top {k}") for k in (10, 50, 100)] + \
              [Line2D([], [], color=INK2, marker="o", label="headline: exactly K neurons, ties broken by body ID"),
               Line2D([], [], color=INK2, marker="o", markerfacecolor=SURFACE, lw=0, label="open marker: a tie at the K-th place (both views shown)"),
               Line2D([], [], color=INK2, marker="+", lw=0, label="+ tie-inclusive: every neuron tied at the K-th value is kept")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    n_tie = {m: tie_info[m] for m in METRICS}
    footer(fig, f"Jaccard = shared neurons / all neurons in either list (0 = no overlap, 1 = identical; the reference compared with itself is 1). Neurons: primary population ({POPN:,}), ranked by neuron-to-neuron out-strength or out-degree. "
                f"Boundary ties are rare here ({n_tie['out_strength_nn']['boundary_tie_rows']} of {n_tie['out_strength_nn']['rows']} strength rows; {n_tie['out_degree_nn']['boundary_tie_rows']} of {n_tie['out_degree_nn']['rows']} degree rows), "
                "so the two views nearly coincide. Each comparison is between two different graphs.", y=-0.075)
    save(fig, "fig4_topk_overlap.png")


# ---------------------------------------------------------------------------------------------------- Figure 5
def fig5(stabs, zero_info):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, (metric, (name, _, _)) in enumerate(METRICS.items()):
        for col, (val, k, title) in enumerate((("jaccard", 50, "Jaccard similarity of the top 50"), ("spearman_full_population", None, "Spearman rank correlation, all neurons"))):
            ax = axes[row, col]; m = fd.pairwise_matrix(stabs[metric], val, k=k, policy="id")
            ax.imshow(m.to_numpy(), cmap=CMAP, vmin=0, vmax=1, origin="upper"); ax.grid(False)
            for i in range(len(THR)):
                for j in range(len(THR)):
                    v = m.iloc[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
            ax.set_xticks(range(len(THR))); ax.set_xticklabels(XL); ax.set_yticks(range(len(THR))); ax.set_yticklabels(XL)
            for s in ax.spines.values(): s.set_visible(False)
            ax.tick_params(length=0); ax.set_title(f"{name}: {title}", loc="left", fontsize=9.5)
            if row == 1: ax.set_xlabel("threshold T")
            if col == 0: ax.set_ylabel("threshold T")
    z = zero_info
    footer(fig, f"Jaccard uses the headline view (exactly K = 50 neurons, ties broken by body ID). Spearman is computed over all {POPN:,} neurons of the fixed population, with average ranks for ties.\n"
                f"Caution for Spearman at high thresholds: neurons with no outgoing neuron-to-neuron connection all have value 0 and tie at the bottom. At T≥50 that is {z['n']:,} of {POPN:,} neurons ({pct(z['share'], 0)}), "
                f"against {z['n1']:,} ({pct(z['share1'], 1)}) at T≥1, so correlations involving T≥50 partly measure which neurons keep any connection, not only the order among strong ones.", y=0.02)
    save(fig, "fig5_pairwise_stability.png")


# ---------------------------------------------------------------------------------------------------- Figure 6
def fig6(trajs):
    fig, axes = plt.subplots(2, 3, figsize=(14, 11.5), sharey=True, gridspec_kw={"hspace": 0.3, "wspace": 0.42})
    x = np.arange(len(THR)); NONE_Y = 3e5
    for row, (metric, (name, _, _)) in enumerate(METRICS.items()):
        t = trajs[metric]
        for col, src in enumerate((1, 10, 50)):
            ax = axes[row, col]; ax.set_yscale("log"); ax.set_ylim(NONE_Y * 1.6, 0.6); ax.set_xlim(-0.35, len(THR) - 1 + 2.1)
            ids = t[t.selected_at.str.split(",").apply(lambda s_: str(src) in s_)].bodyId.unique()
            ends = []
            for b in ids:
                g = t[t.bodyId == b].sort_values("threshold"); c, mk = SUPER_STYLE.get(g.superclass.iloc[0], (MUTED, "o"))
                y = g["rank"].to_numpy(); has = g.has_surviving_outgoing.to_numpy()
                ax.plot(x[has], y[has], color=c, marker=mk, ms=4.0, lw=1.1, alpha=0.95, zorder=3)
                if (~has).any():
                    ax.scatter(x[~has], np.full((~has).sum(), NONE_Y), marker="x", s=26, color=c, linewidths=1.2, zorder=4)
                last = np.flatnonzero(has)[-1]
                ends.append((x[last], y[last], f"{g.type.iloc[0]} {int(b)}", c))
            ax.axhline(2.0e5, color=MUTED, lw=0.7, ls=(0, (2, 3)), zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(XL)
            ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5, NONE_Y]); ax.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000", "no\nrank"])
            ax.set_title(f"{name}: top 10 at T≥{src} ({len(ids)} neurons)", loc="left", fontsize=9.2)
            if col == 0: ax.set_ylabel(f"rank among all {POPN:,} neurons (1 = highest)")
            if row == 1: ax.set_xlabel("threshold T")
            place_end_labels(ax, ends, len(THR) - 1 + 0.4, min_gap_pt=8.6, fontsize=6.8)
    handles = [Line2D([], [], color=c, marker=m, label=s_.replace("_", " ")) for s_, (c, m) in SUPER_STYLE.items()] + \
              [Line2D([], [], color=INK2, marker="x", lw=0, label="× = no surviving outgoing neuron connection (no rank assigned)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.045), fontsize=8)
    footer(fig, "Selection rule (fixed in advance): every neuron in the top 10 at T≥1, T≥10 or T≥50; a neuron selected at several thresholds appears in each matching panel. Lines follow each neuron’s rank at all six thresholds. "
                f"Rank = position among all {POPN:,} neurons by value (descending), ties broken by body ID; a neuron with value 0 has no rank. Labels: type and body ID. High rank here is a structural measure, not evidence of importance.", y=-0.005)
    save(fig, "fig6_rank_trajectories.png")


# ---------------------------------------------------------------------------------------------------- Figure 7 (optional)
def fig7(agg, mem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.22})
    cols = ["#3b2a6b", "#5c46a0", "#7b62b8", "#9a85cc", "#b9a8dc", "#d7ccec"]
    for ax, (metric, (name, key, desc)) in zip(axes, METRICS.items()):
        for ti, T in enumerate(THR):
            xs, ns = fd.ccdf_counts(agg[f"per_body_{POP}"][ti, M[key]][mem])
            ax.step(xs, ns, where="post", color=cols[ti], lw=1.4, label=f"T≥{T}")
        ax.axhline(100, color=MUTED, lw=0.8, ls=(0, (4, 3))); ax.text(1.05, 118, "100 neurons", fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.8, 3e5)
        ax.set_xlabel(f"{name.lower()} to neuron partners ({'summed weight' if key == 'out_str_n' else 'distinct partners'})"); ax.set_ylabel("number of neurons with at least this value")
        ax.set_title(name, loc="left"); ax.legend(loc="upper right", ncol=2)
    footer(fig, f"Primary population; only neurons with a positive value are drawn (the others are counted in Figure 2). The horizontal dashed line marks 100 neurons: where each curve crosses it is the value of the 100th-ranked neuron. "
                "Descriptive only; no claim is made about the functional form of these distributions.", y=-0.01)
    save(fig, "fig7_value_distributions.png")


# ---------------------------------------------------------------------------------------------------- captions
def write_readme(ctx):
    r, sh, cat, tie, z = ctx["ret"], ctx["shares"], ctx["cat"], ctx["tie"], ctx["zero"]
    g = lambda T, col: r[r.threshold == T][col].iloc[0]
    nn_rec = {T: cat[(cat.threshold == T) & (cat.category == "NN")].records_share.iloc[0] for T in THR}
    nn_w = {T: cat[(cat.threshold == T) & (cat.category == "NN")].weight_share.iloc[0] for T in THR}
    j = ctx["jref"]
    lines = [
        "# Figures and captions", "",
        "Derived from **MaleCNS v1.0** (FlyEM Project Team at HHMI Janelia, Cambridge Drosophila Connectomics Group / MRC LMB, Google Research), licensed CC BY 4.0 "
        "(<https://creativecommons.org/licenses/by/4.0/>); source <https://male-cns.janelia.org/download/>; preprint doi:10.1101/2025.10.09.680999. "
        "The data files were used unmodified; these figures are derived summaries. The MIT licence of this repository does not cover the data.", "",
        "## Definitions used in every figure", "",
        "- **Threshold T:** keep a connection record when its weight is at least T (≥). Thresholds: 1, 2, 5, 10, 20, 50. Weight = number of synaptic connection records between a pair of bodies, per the data providers’ code (the dataset documentation does not define it in prose).",
        f"- **Primary population:** bodies with a `superclass` annotation ({POPN:,}). Provisional: it rests on the annotation semantics given in the preprint. The population is fixed across thresholds, including neurons left with no connection.",
        f"- **Fully traced subset (sensitivity analysis):** `superclass` present **and** `status == \"Traced\"` ({ctx['n_traced']:,} bodies). It is a sensitivity analysis, not a superior biological definition; it excludes many sensory bodies labelled “Out of scope” ({ctx['n_oos']:,} `ol_sensory` bodies among the {POPN - ctx['n_traced']:,} it drops). Its top lists are identical or almost identical to the primary population’s, so the figures show the primary population.",
        "- **Edge categories (source → target):** N = a body of the primary population; **O = “other body” = every body not in that population** (annotated bodies without a superclass and all unannotated bodies). NN, NO, ON, OO.",
        "- **Metrics:** out-strength = summed weight of outgoing records to neuron partners; out-degree = number of distinct downstream neuron partners (each record is a distinct pair; no duplicate pairs exist).",
        "- **Ties:** headline top-K sets break ties by body ID (exactly K neurons); the tie-inclusive view keeps every neuron tied at the K-th value. Both are always shown; boundary ties are flagged.",
        f"- The 166,691 (preprint) versus {POPN:,} (this primary population) count difference is unresolved and is not adjusted for. Findings apply to this single dataset version and these graph definitions. Rank is a structural measure, not evidence of biological importance.", "",
        "## Figure 1 – weight distribution (`fig1_weight_distribution.png`)", "",
        f"Number of body pairs at each connection weight (log–log), with the six thresholds. Shares of records kept: " + ", ".join(f"T≥{T}: {pct(g(T, 'records_share'), 2)}" for T in THR) +
        ". Computed from `threshold_totals.csv` and checked against the weight histogram. Descriptive; no functional form is claimed.", "",
        "## Figure 2 – what survives (`fig2_retention.png`)", "",
        f"(a) Share of records and of summed weight kept (log scale): at T≥50, {pct(g(50, 'records_share'), 2)} of records but {pct(g(50, 'weight_share'), 1)} of the weight. "
        f"(b) Share of the fixed {int(sh.population_size.iloc[0]):,} primary neurons with an outgoing, incoming, any, or no neuron-to-neuron connection: any = " + ", ".join(f"{pct(v, 0)}" for v in sh['any']) +
        " from T≥1 to T≥50; none = " + ", ".join(f"{pct(v, 0)}" for v in sh['none']) + ".", "",
        "## Figure 3 – edge-category composition (`fig3_edge_categories.png`)", "",
        "Share of retained records and of summed weight in each category (primary population). Neuron-to-neuron share of records: " + ", ".join(f"T≥{T}: {pct(nn_rec[T], 0)}" for T in THR) +
        "; of weight: " + ", ".join(f"T≥{T}: {pct(nn_w[T], 0)}" for T in THR) + ". Raising the threshold therefore also changes the composition of the graph.", "",
        "## Figure 4 – top-K overlap (`fig4_topk_overlap.png`)", "",
        "Jaccard similarity of the top 10 / 50 / 100 neurons at each threshold against the T≥1 list (top row) and the T≥10 list (bottom row), for out-strength (left) and out-degree (right); filled markers and lines are the headline view, “+” the tie-inclusive view, open markers a tie at the K-th place. "
        f"Against T≥1, top-10 Jaccard at T≥10 is {j['out_strength_nn'][1][10][10]:.2f} for strength and {j['out_degree_nn'][1][10][10]:.2f} for degree; top-100: {j['out_strength_nn'][1][10][100]:.2f} and {j['out_degree_nn'][1][10][100]:.2f}. "
        f"Boundary ties: {tie['out_strength_nn']['boundary_tie_rows']} of {tie['out_strength_nn']['rows']} strength rows and {tie['out_degree_nn']['boundary_tie_rows']} of {tie['out_degree_nn']['rows']} degree rows (at most {max(tie['out_strength_nn']['max_tied_at_kth_value'], tie['out_degree_nn']['max_tied_at_kth_value'])} neurons share the K-th value).", "",
        "## Figure 5 – all-pairs stability (`fig5_pairwise_stability.png`)", "",
        f"Jaccard (top 50, headline view) and Spearman rank correlation over all {POPN:,} neurons for every pair of thresholds. Spearman caution: neurons with no outgoing neuron-to-neuron connection tie at value 0 — {z['n']:,} neurons ({pct(z['share'], 0)}) at T≥50 versus {z['n1']:,} ({pct(z['share1'], 1)}) at T≥1 — so correlations involving high thresholds partly reflect which neurons keep any connection.", "",
        "## Figure 6 – rank trajectories (`fig6_rank_trajectories.png`)", "",
        "Rank of every neuron that is in the top 10 at T≥1, T≥10 or T≥50, followed across all six thresholds (rows: strength, degree; columns: the threshold that selected them; a neuron selected at several thresholds appears in each). Neurons with no surviving outgoing neuron connection are marked × and given no rank. "
        "The underlying table, with selection source, type, superclass, rank and survival flag per neuron and threshold, is `outputs/tables/rank_trajectories_<metric>.csv`.", "",
    ]
    if ctx.get("fig7"):
        lines += ["## Figure 7 – value distributions (`fig7_value_distributions.png`)", "", ctx["fig7_text"], ""]
    lines += ["Palette: light mode, validated with a colour-palette checker (lightness band, chroma floor, adjacent-pair colour-blind separation, normal-vision floor, contrast on the chart surface all pass); identity is also carried by markers and labels.", ""]
    with open(f"{F_OUT}/README.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote", f"{F_OUT}/README.md")


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-fig7", action="store_true")
    ap.add_argument("--agg", default="data/interim/aggregation/aggregates.npz"); ap.add_argument("--annot", default="data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    a = ap.parse_args(); os.makedirs(F_OUT, exist_ok=True)
    totals = pd.read_csv(f"{T_OUT}/threshold_totals.csv"); hist = pd.read_csv(f"{T_OUT}/weight_histogram.csv"); edge = pd.read_csv(f"{T_OUT}/edge_category_totals.csv")
    popt = pd.read_csv(f"{T_OUT}/population_by_threshold.csv"); sup = pd.read_csv(f"{T_OUT}/ranking_support.csv")
    probs = fd.verify_sources(totals, hist, edge, popt, sup)
    if probs:
        print("INPUT TABLES DISAGREE:\n  " + "\n  ".join(probs)); sys.exit(1)
    print("input tables cross-checked: consistent")
    stabs = {m: pd.read_csv(f"{T_OUT}/stability_{POP}_{m}.csv") for m in METRICS}
    ties = {m: pd.read_csv(f"{T_OUT}/ties_{POP}_{m}.csv") for m in METRICS}
    tops = {m: pd.read_csv(f"{T_OUT}/top100_{POP}_{m}.csv") for m in METRICS}
    global POPN
    ret = fd.retention(totals); shares = fd.population_shares(popt, POP); cat = fd.category_shares(edge, POP)
    POPN = int(shares.population_size.iloc[0])
    tie_info = {m: fd.tie_summary(ties[m]) for m in METRICS}
    zs = fd.zero_valued_counts(sup, POP, "out_strength_nn").set_index("threshold")
    zero_info = dict(n=int(zs.loc[50, "zero_valued"]), share=float(zs.loc[50, "zero_share"]), n1=int(zs.loc[1, "zero_valued"]), share1=float(zs.loc[1, "zero_share"]))
    jref = {m: {ref: {k: {} for k in (10, 50, 100)} for ref in (1, 10)} for m in METRICS}
    for m in METRICS:
        for ref in (1, 10):
            d = fd.jaccard_vs_reference(stabs[m], ref)
            for _, r_ in d[d.tie_policy == "id"].iterrows():
                jref[m][ref][int(r_.k)][int(r_.threshold)] = r_.jaccard
    # the caption helper indexes j[metric][ref=1][threshold=10][k]; reshape accordingly
    jc = {m: {1: {10: {k: jref[m][1][k][10] for k in (10, 50, 100)}}} for m in METRICS}

    # ---- rank trajectories (reproducible table behind Figure 6)
    agg = np.load(a.agg); ann = load_annotations(a.annot); mem = make_populations(ann)[POP]
    assert np.array_equal(agg["ann_ids"], ann.bodyId.to_numpy()), "aggregates were built from a different annotation table"
    ids = ann.bodyId.to_numpy()[mem]; meta = ann.loc[mem, ["type", "superclass"]].reset_index(drop=True)
    trajs = {}
    for m, (_, key, _) in METRICS.items():
        vals = {T: agg[f"per_body_{POP}"][ti, M[key]][mem] for ti, T in enumerate(THR)}
        sel = fd.select_trajectory_neurons(tops[m], thresholds=(1, 10, 50), n=10)
        trajs[m] = fd.trajectory_table(vals, ids, sel, meta)
        trajs[m].to_csv(f"{T_OUT}/rank_trajectories_{m}.csv", index=False)
        print(f"{m}: {len(sel)} selected neurons -> rank_trajectories_{m}.csv")
        # consistency: a trajectory rank inside the top 100 must equal the rank in the saved top100 table
        chk = trajs[m].dropna(subset=["rank"]).merge(tops[m][["threshold", "rank", "bodyId"]].rename(columns={"rank": "rank_saved"}), on=["threshold", "bodyId"])
        assert (chk["rank"] == chk.rank_saved).all(), "trajectory ranks disagree with the saved top-100 tables"

    fig1(ret, hist); fig2(ret, shares); fig3(cat); fig4(stabs, tie_info); fig5(stabs, zero_info); fig6(trajs)
    traced_n = int(popt[popt.population == "sensitivity"].population_size.iloc[0])
    oos = ann[ann.superclass.notna() & (ann.status.astype(object) != "Traced") & (ann.superclass == "ol_sensory") & (ann.statusLabel.astype(object) == "Out of scope")]
    ctx = dict(ret=ret, shares=shares, cat=cat, tie=tie_info, zero=zero_info, jref=jc, fig7=False, n_traced=traced_n, n_oos=int(len(oos)))
    if not a.no_fig7:
        fig7(agg, mem); ctx["fig7"] = True
        kth = lambda m, T: int(ties[m][(ties[m].threshold == T) & (ties[m].k == 100)].kth_value.iloc[0])
        ctx["fig7_text"] = ("Number of primary neurons with at least a given neuron-to-neuron out-strength (left) or out-degree (right), one curve per threshold (positive values only). "
                            f"The value of the 100th-ranked neuron (where each curve crosses 100 neurons) is, for T≥1 / T≥10 / T≥50: out-strength {kth('out_strength_nn', 1):,} / {kth('out_strength_nn', 10):,} / {kth('out_strength_nn', 50):,}; "
                            f"out-degree {kth('out_degree_nn', 1):,} / {kth('out_degree_nn', 10):,} / {kth('out_degree_nn', 50):,}. Between T≥1 and T≥50 the 100th-ranked value falls {kth('out_strength_nn', 1) / kth('out_strength_nn', 50):.1f}-fold for out-strength but {kth('out_degree_nn', 1) / kth('out_degree_nn', 50):.0f}-fold for out-degree, so the upper tail of out-strength moves comparatively little with the threshold while the out-degree distribution shifts a great deal, "
                            "which is consistent with (but does not by itself prove) the lower stability of degree-based lists in Figure 4. Descriptive; no functional form is claimed.")
    write_readme(ctx)


if __name__ == "__main__":
    main()
