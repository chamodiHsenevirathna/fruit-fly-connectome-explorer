"""The numbers quoted in docs/report.md and README.md must match the saved tables, and every file they point to must exist."""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
REPORT = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
THR = (1, 2, 5, 10, 20, 50)


def _csv(name):
    return pd.read_csv(T / name)


def test_threshold_table_rows_match_outputs():
    tt = _csv("threshold_totals.csv").set_index("threshold"); pop = _csv("population_by_threshold.csv")
    pop = pop[pop.population == "primary"].set_index("threshold"); ec = _csv("edge_category_totals.csv"); ec = ec[ec.population == "primary"]
    base = tt.loc[1]
    for t in THR:
        e = ec[ec.threshold == t]; nn_share = 100 * e[e.category == "NN"].records.iloc[0] / e.records.sum()
        n = pop.loc[t, "population_size"]
        row = (f"| {t} | {int(tt.loc[t, 'records']):,} | {100 * tt.loc[t, 'records'] / base.records:.2f} | {100 * tt.loc[t, 'strength'] / base.strength:.2f} | "
               f"{100 * pop.loc[t, 'neurons_with_any_NN_connection'] / n:.2f} | {100 * pop.loc[t, 'neurons_with_any_NN_out'] / n:.2f} | {nn_share:.1f}% |")
        assert row in REPORT, row


def test_jaccard_and_spearman_rows_match_outputs():
    for label, m in (("out-strength", "out_strength_nn"), ("out-degree", "out_degree_nn")):
        s = _csv(f"stability_primary_{m}.csv"); head = s[(s.tie_policy == "id") & (s.threshold_a == 1)]
        for k in (10, 50, 100):
            vals = [head[(head.threshold_b == t) & (head.k == k)].jaccard.iloc[0] for t in THR[1:]]
            row = f"| {label} | {k} | " + " | ".join(f"{v:.3f}" for v in vals) + " |"
            assert row in REPORT, row
        rho = [s[(s.threshold_a == 1) & (s.threshold_b == t)].spearman_full_population.iloc[0] for t in THR[1:]]
        assert ", ".join(f"{v:.3f}" for v in rho) in REPORT


def test_zero_valued_counts_and_value_of_100th_neuron():
    sup = _csv("ranking_support.csv"); sup = sup[sup.population == "primary"]
    for m in ("out_strength_nn", "out_degree_nn"):
        z = sup[sup.metric == m].sort_values("threshold"); zeros = (z.population_size - z.neurons_with_positive_value).tolist()
        if m == "out_strength_nn":
            first = zeros
            for c, n in zip(zeros, z.population_size):
                assert f"{c:,} ({100 * c / n:.1f}%)" in REPORT
        else:
            assert zeros == first                                              # report says the counts are identical for both metrics
        top = _csv(f"top100_primary_{m}.csv"); v = top[top["rank"] == 100].set_index("threshold").value
        assert f"{v[1]:,}" in REPORT and f"{v[50]:,}" in REPORT
        assert f"a factor of {v[1] / v[50]:.1f}" in REPORT or f"a factor of {v[1] / v[50]:.0f}" in REPORT


def test_fully_traced_subset_claims():
    pops = _csv("population_by_threshold.csv").groupby("population").population_size.first()
    assert f"**{pops['primary']:,}**" in REPORT and f"**{pops['sensitivity']:,}**" in REPORT
    worst = {}
    for m in ("out_strength_nn", "out_degree_nn"):
        a = _csv(f"stability_primary_{m}.csv"); b = _csv(f"stability_sensitivity_{m}.csv")
        j = a.merge(b, on=["threshold_a", "threshold_b", "k", "tie_policy"], suffixes=("_p", "_s"))
        assert len(j) == 90 and (j.jaccard_p - j.jaccard_s).abs().max() == 0
        worst[m] = (j.spearman_full_population_p - j.spearman_full_population_s).abs().max()
    assert f"{worst['out_strength_nn']:.4f} (out-strength) and {worst['out_degree_nn']:.4f} (out-degree)" in REPORT


def test_tie_sensitivity_claims():
    out = {}
    for pop in ("primary", "sensitivity"):
        for m in ("out_strength_nn", "out_degree_nn"):
            out[pop, m] = _csv(f"tie_sensitivity_{pop}_{m}.csv")
    assert len(out["primary", "out_strength_nn"]) == 0 and len(out["sensitivity", "out_strength_nn"]) == 0
    d = out["primary", "out_degree_nn"]; diff = d.id_minus_mc_mean
    assert len(d) == 14 and (diff >= 1e-9).sum() == 8 and (diff <= -1e-9).sum() == 2 and (diff.abs() < 1e-9).sum() == 4
    assert d.groupby("k").size().to_dict() == {50: 5, 100: 9}
    assert f"{diff.abs().max():.4f}" == "0.0070" and d.mc_sem.max() < 1.5e-4 and set(d.n_draws) == {10000}
    assert out["sensitivity", "out_degree_nn"][["threshold_a", "threshold_b", "k"]].equals(d[["threshold_a", "threshold_b", "k"]])
    assert (out["sensitivity", "out_degree_nn"].id_minus_mc_mean - diff).abs().max() < 1e-12
    for needle in ("**0 of 45**", "**14 of 45**", "**higher than**", "in 8,", "**lower** in 2", "**equal** in 4", "**0.0070**", "10,000"):
        assert needle in REPORT, needle
    r = d[(d.threshold_a == 20) & (d.threshold_b == 50) & (d.k == 50)].iloc[0]
    assert f"headline {r.jaccard_id_deterministic:.4f}, random mean {r.mc_mean:.4f}" in REPORT
    r = d[(d.threshold_a == 5) & (d.threshold_b == 10) & (d.k == 100)].iloc[0]
    assert f"{r.jaccard_id_deterministic:.4f} against a random mean of {r.mc_mean:.4f}" in REPORT


def test_data_facts_match_validation_summary():
    import json
    v = json.loads((T / "validation_summary.json").read_text())
    for needle in (f"{v['rows']:,}", f"{v['distinct_any']:,}", f"{v['weight_sum']:,}", f"{v['weight_max']:,}", f"{v['self_loops']}", f"{v['annotated_bodies']:,}",
                   f"{v['rows_with_any_unmatched']:,} records ({100 * v['rows_with_any_unmatched'] / v['rows']:.1f}%)"):
        assert needle in REPORT, needle
    assert v["extra_duplicate_records"] == 0 and v["nulls"] == 0


def test_readme_headlines_match_outputs():
    tt = _csv("threshold_totals.csv").set_index("threshold"); pop = _csv("population_by_threshold.csv"); pop = pop[pop.population == "primary"].set_index("threshold")
    sp = _csv("stability_primary_out_strength_nn.csv"); dg = _csv("stability_primary_out_degree_nn.csv")
    g = lambda s, t: s[(s.tie_policy == "id") & (s.threshold_a == 1) & (s.threshold_b == t) & (s.k == 100)].jaccard.iloc[0]
    sup = _csv("ranking_support.csv"); z = sup[(sup.population == "primary") & (sup.metric == "out_strength_nn") & (sup.threshold == 50)].iloc[0]
    for needle in (f"**{100 * tt.loc[50, 'records'] / tt.loc[1, 'records']:.2f}%**".replace("0.15", "0.15"), f"**{100 * tt.loc[50, 'strength'] / tt.loc[1, 'strength']:.2f}%**",
                   f"**{100 * pop.loc[50, 'neurons_with_any_NN_connection'] / pop.loc[50, 'population_size']:.1f}%**", f"{g(sp, 2):.3f} at T≥2 and {g(sp, 50):.3f} at T≥50",
                   f"{g(dg, 2):.3f} at T≥2 and {g(dg, 50):.3f} at T≥50", f"{100 * (z.population_size - z.neurons_with_positive_value) / z.population_size:.1f}%", "0.0070", "14 of 45"):
        assert needle in README, needle


def test_referenced_files_and_links_exist():
    for doc_path in (ROOT / "README.md", ROOT / "docs" / "report.md"):
        text = doc_path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):                  # markdown links and images
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc_path.parent / target).resolve().exists(), f"{doc_path.name}: broken link {target}"
        for ref in set(re.findall(r"`((?:scripts|src|tests|outputs|docs)/[A-Za-z0-9_./-]+)`", text)):   # backticked repository paths
            if "*" in ref or ref.endswith("/"):
                continue
            assert (ROOT / ref).exists(), f"{doc_path.name}: missing file {ref}"
        for script in set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", text)):
            assert (ROOT / "scripts" / script).exists(), script


def test_public_docs_do_not_point_to_private_notes():
    # The private documents are the ones .gitignore keeps out of the repository under docs/; none may be named in a public document.
    private = [l.strip().rstrip("/") for l in (ROOT / ".gitignore").read_text().splitlines() if l.strip().startswith("docs/")]
    private += [Path(p).stem for p in private if p.endswith(".md")]
    private += [f.stem for f in (ROOT / "docs" / "internal").glob("*.md")]       # local-only notes, when present
    assert private, "expected docs/ entries in .gitignore"
    for doc in (README, REPORT, (ROOT / "outputs" / "README.md").read_text(encoding="utf-8"), (ROOT / "outputs" / "figures" / "README.md").read_text(encoding="utf-8")):
        for name in private:
            assert name not in doc, name
