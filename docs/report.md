# How sensitive is neuron ranking to the connection-weight threshold? A study of the MaleCNS v1.0 connectome

**Scope.** One dataset version (MaleCNS v1.0, minconf-0.5 flat connectome), six thresholds, two connectivity metrics. All numbers below are read from the tables in [`outputs/tables/`](../outputs/tables/); the tests in `tests/test_report_numbers.py` check them against those tables. Observations (what the tables show) are kept separate from interpretations (what they might mean).

## Summary
- Raising the weight threshold from 1 to 50 keeps 0.15% of connection records but 6.56% of the summed weight, and leaves 39.9% of the primary population with any neuron-to-neuron connection.
- The top-ranked neurons by **out-strength** (summed weight to neuron partners) change moderately: the top-100 Jaccard similarity against the T≥1 list is 0.905 at T≥2 and 0.342 at T≥50.
- The top-ranked neurons by **out-degree** (number of distinct neuron partners) change much more: 0.562 at T≥2 and 0.064 at T≥50.
- Random tie-breaking changes the top-k Jaccard by at most 0.0070 in the comparisons where a boundary tie exists.
- These are structural results about ranking by connectivity. They say nothing about which neurons matter biologically.

## 1. Question and motivation
Analyses of connectomes often discard weak connections before ranking neurons, for example by out-going connectivity. The cut-off is usually a convenience, yet it decides which graph is analysed. This project asks: **how sensitive is the identification of highly ranked neurons to the connection-weight threshold?** Because each threshold produces a *different graph*, rankings at different thresholds are rankings of different objects. The question is therefore whether the same neurons stay near the top as that graph is thinned, not whether one ranking is "right".

## 2. Data: provenance, licence and graph construction
- **Source.** MaleCNS v1.0, released by the FlyEM Project Team at HHMI Janelia, the Cambridge Drosophila Connectomics Group (University of Cambridge) / MRC Laboratory of Molecular Biology, and Google Research (<https://male-cns.janelia.org/download/>). **Licence: CC BY 4.0.** Files used, unmodified: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1,051,241,946 bytes) and `body-annotations-male-cns-v1.0-minconf-0.5.feather`.
- **Graph.** The table has 151,856,684 directed body pairs (no nulls, no duplicate pairs, 123 self-loops), 88,384,522 distinct bodies, and weights from 1 to 2,591 summing to 311,833,243 (`validation_summary.json`, `duplicate_check.json`).
- **Meaning of `weight`.** The dataset's prose documentation does not define it. This project reads it as the number of synaptic connection records between the pair, following the data providers' published code and consistency checks (the weights sum to 311,833,243, equal to the `totalPostCount` field of the dataset's neuPrint import metadata, `Neuprint_Meta.csv` under `gs://flyem-male-cns/v1.0/database/neuprint-inputs/`). It is a count, not a measure of physiological strength. `minconf-0.5` means both the pre- and postsynaptic site have confidence at least 0.5.
- **Endpoints are bodies, not neurons.** Only 211,577 of the 88.4 million bodies carry annotations. 125,828,298 records (82.9%) have at least one unannotated endpoint.

## 3. Populations
- **Primary population:** bodies with a `superclass` annotation, **166,700** bodies.
- **Fully traced subset (sensitivity analysis):** `superclass` present *and* `status == "Traced"`, **164,606** bodies. It is a sensitivity analysis, not a better definition of a neuron: it also drops many sensory bodies labelled "Out of scope" (counts in [`outputs/figures/README.md`](../outputs/figures/README.md)).
- Both definitions are provisional. The preprint reports 166,691 neurons against 166,700 here; the reason for that difference is unresolved and nothing is adjusted for it.
- Each population is **fixed across thresholds**, including neurons that lose all their connections, so rankings at different thresholds cover the same neurons.

## 4. Thresholds
A record is kept when `weight >= T`, for T in {1, 2, 5, 10, 20, 50}; T≥1 keeps everything. Edges are classed by their endpoints: NN (neuron to neuron), NO, ON, OO, where "other" is any body outside the population.

| T | records kept | % of records | % of summed weight | % of primary with any NN connection | % with an outgoing NN connection | NN share of records |
|---|---|---|---|---|---|---|
| 1 | 151,856,684 | 100.00 | 100.00 | 99.87 | 99.27 | 16.8% |
| 2 | 57,670,765 | 37.98 | 69.80 | 99.79 | 98.51 | 26.5% |
| 5 | 7,622,864 | 5.02 | 31.42 | 99.48 | 97.29 | 81.9% |
| 10 | 2,799,910 | 1.84 | 21.78 | 98.23 | 93.72 | 98.4% |
| 20 | 1,068,256 | 0.70 | 14.42 | 87.24 | 75.84 | 99.6% |
| 50 | 228,233 | 0.15 | 6.56 | 39.90 | 29.48 | 99.9% |

*Observation:* the composition of the graph changes as well as its size. The neuron-to-neuron share of records rises from 16.8% to 99.9%, and of summed weight from 39.8% to 99.9%.

## 5. Metrics and ranking-stability methods
- **Out-strength (NN):** summed weight of a neuron's outgoing records to neuron partners. **Out-degree (NN):** number of distinct downstream neuron partners.
- **Top-k lists** for k = 10, 50, 100. **Jaccard similarity** of two lists = |intersection| / |union| (0 = disjoint, 1 = identical).
- **Spearman rank correlation** is computed over the whole fixed population with average ranks for ties.
- **Ties.** Many neurons share a value, especially for degree. Two views are always reported: the *headline* view breaks ties by body ID and returns exactly k neurons; the *tie-inclusive* view keeps every neuron tied at the k-th value. A **boundary tie** means the k-th and (k+1)-th values are equal, so the headline list depends on the arbitrary tie-break.
- **Zero-valued neurons.** A neuron with no surviving outgoing NN record has value 0 and receives no rank in trajectories, and tied zeros are never admitted to the tie-inclusive set. In Spearman they tie at the bottom (below).
- **Verification.** The streaming aggregation was checked against an independent pandas reference on real data slices (zero mismatches, `aggregation_crosscheck.log`), with tests that fail when the code is deliberately broken. Separately, a sample check (`body_stats_crosscheck.log`, 500,746 rows) found that per-body summed in- and out-weight matched the dataset's own per-body statistics table for every sampled body; that table's degree-like columns agreed for only 71.5% of bodies, their definition was not established, and they are not used as a check.

## 6. Findings (observations)
**Jaccard similarity of the top-k list at each threshold against the T≥1 list** (primary population, headline view):

| Metric | k | T≥2 | T≥5 | T≥10 | T≥20 | T≥50 |
|---|---|---|---|---|---|---|
| out-strength | 10 | 1.000 | 0.667 | 0.667 | 0.429 | 0.176 |
| out-strength | 50 | 0.887 | 0.786 | 0.613 | 0.587 | 0.389 |
| out-strength | 100 | 0.905 | 0.770 | 0.639 | 0.515 | 0.342 |
| out-degree | 10 | 0.429 | 0.250 | 0.111 | 0.111 | 0.000 |
| out-degree | 50 | 0.562 | 0.316 | 0.149 | 0.124 | 0.064 |
| out-degree | 100 | 0.562 | 0.351 | 0.170 | 0.099 | 0.064 |

**Spearman correlation against T≥1 (all primary neurons):** out-strength 0.996, 0.957, 0.885, 0.768, 0.599 and out-degree 0.969, 0.818, 0.651, 0.481, 0.349 for T≥2, 5, 10, 20, 50.

**Tied zeros at high thresholds.** The number of neurons with no outgoing NN connection, which all share the value 0, is 1,220 (0.7%) at T≥1, 2,484 (1.5%), 4,521 (2.7%), 10,473 (6.3%), 40,270 (24.2%) and **117,564 (70.5%) at T≥50**, identical for both metrics. Spearman correlations involving high thresholds therefore partly reflect *which* neurons keep any connection, not only how the survivors are ordered.

**Size of the top of the distribution.** The value of the 100th-ranked neuron falls from 18,662 (T≥1) to 6,600 (T≥50) for out-strength (a factor of 2.8), and from 3,419 to 60 for out-degree (a factor of 57).

**Fully traced subset.** Jaccard values are identical to the primary population in all 90 comparisons per metric (two tie views included); Spearman values differ by at most 0.0077 (out-strength) and 0.0091 (out-degree).

## 7. Figures
Captions and definitions: [`outputs/figures/README.md`](../outputs/figures/README.md).

| Figure | Shows | Interpretation limit |
|---|---|---|
| 1 `fig1_weight_distribution` | Body pairs at each weight, with the thresholds | Descriptive; no functional form claimed |
| 2 `fig2_retention` | Records, weight and neurons kept | Section 4 table |
| 3 `fig3_edge_categories` | NN/NO/ON/OO composition | The graph changes in kind, not only in size |
| 4 `fig4_topk_overlap` | Top-k Jaccard against T≥1 and T≥10 | Both tie views and boundary ties shown |
| 5 `fig5_pairwise_stability` | All threshold pairs, Jaccard and Spearman | Spearman affected by tied zeros |
| 6 `fig6_rank_trajectories` | Ranks of neurons that top the list at T≥1, 10 or 50 | Neurons without surviving connections get no rank |
| 7 `fig7_value_distributions` | Distribution of out-strength and out-degree | Consistent with, not proof of, the stability gap |

## 8. Tie-breaking sensitivity (Monte Carlo)
**Method.** For every comparison with a boundary tie in at least one list, neurons strictly above the k-th value are always included, and the open slots are filled by a uniformly random subset of the neurons tied at the k-th value, drawn independently for the two thresholds. This is repeated 10,000 times with a seeded generator (`scripts/run_tie_sensitivity.py`; each row's seed is derived from a base seed, the thresholds and k). Where neither list has a boundary tie, the top-k set cannot depend on the tie-break, so nothing is sampled.

**Results (identical for both populations).**
- Out-strength: **0 of 45** threshold-pair comparisons have a boundary tie.
- Out-degree: **14 of 45** do (5 at k = 50, 9 at k = 100, none at k = 10).
- In those 14, the deterministic headline Jaccard is **higher than** the random-tie-breaking mean in 8, **lower** in 2 and **equal** in 4 (difference below 1e-9; the tied neurons cannot change the intersection).
- The largest gap is **0.0070** (T≥20 versus T≥50, k = 50: headline 0.4493, random mean 0.4423). In the other direction, T≥5 versus T≥10 at k = 100 gives 0.5385 against a random mean of 0.5444.
- The Monte Carlo standard error of the mean is at most 0.0001.

**Reading these results.**
- The standard error is **sampling uncertainty of the estimate** from using 10,000 draws. It is not the overall scientific uncertainty, which also includes the provisional neuron definition, the meaning of `weight`, and the choice of dataset version. The values are estimates, not exact results; against exact enumeration on all 28 real rows the largest difference was 0.0001 (a one-off check, not part of the saved outputs).
- The data do **not** show that deterministic tie-breaking consistently inflates similarity: the sign of the difference varies. A published result on rank-biased overlap states that deterministic tie-breaking (for example by document ID) inflates similarity there, because tied items appear in the same order in both rankings (Corsi and Urbano 2024). Whether that holds for top-k Jaccard depends on the tied groups, and in this dataset it does not hold uniformly.
- The check covers ties at the k-th place only, treats the order of tied neurons as unknown, and says nothing about which tied neuron matters more. The gaps are small next to the differences between thresholds (Section 6).

## 9. Interpretation (not directly tested here)
- Out-degree rankings may be less stable because the distribution of degree at the top shrinks much more with the threshold (Section 6, Figure 7). This is consistent with the data but is not a tested causal explanation.
- A neuron's high rank under a given metric and threshold means it has high connectivity in that graph. It does not show that the neuron is important for behaviour or function, and a stable rank does not show a causal role.

## 10. Limitations
- Different thresholds describe different graphs; rankings are compared across them only to measure sensitivity.
- `weight` is a record count whose definition rests on the providers' code, not on prose documentation.
- Endpoints are bodies; 82.9% of records involve an unannotated body, whose nature is not examined here. The neuron definitions are provisional and annotation coverage is incomplete.
- Only two metrics, six thresholds and k in {10, 50, 100} were examined, on one dataset version and one confidence cut-off.
- Tied zero-valued neurons affect Spearman at high thresholds (Section 6). Boundary ties are rare for these data but present for out-degree.
- Run time and memory were measured on one machine only: two aggregation runs took 78 s (peak 845 MB, saved in `aggregation_run.json`) and 79 s (peak 755 MB).

## 11. Reproducibility
Python 3.14.7, pandas 3.0.6, pyarrow 26.0.0, numpy 2.5.4 and matplotlib 3.11.2 were used and tested; `pyproject.toml` declares Python 3.10 or later and older minimum package versions, which have not been tested. The pipeline is: download the two data files, `validate_connections.py` (full scan, optional), `run_aggregation.py`, `run_stability.py`, `run_tie_sensitivity.py`, `make_figures.py`, then `pytest`. Exact commands are in the [README](../README.md#reproduce-the-results). The data are not stored in the repository. Before publication, the aggregation, stability and tie-sensitivity steps were re-run into a separate folder: 20 of the 21 regenerated tables were byte-identical to the saved ones (the 21st, `aggregation_run.json`, differs only in its timing and memory record), and the figures and captions regenerated from them were byte-identical to the saved files. This holds for the same software versions and does not by itself establish identical results on other platforms.

## 12. Conclusions
1. In MaleCNS v1.0, which neurons rank highest by out-degree depends strongly on the weight threshold (top-100 Jaccard 0.064 at T≥50 against T≥1), and by out-strength depends on it moderately (0.342).
2. High thresholds discard almost all records (99.85% at T≥50) while keeping a larger share of the weight (6.56%), and leave most primary neurons with no outgoing neuron-to-neuron connection (70.5%).
3. Tie-breaking at the k-th place is a minor source of variation here (at most 0.0070 in Jaccard).
4. These conclusions are about structural rankings in one dataset version under provisional neuron definitions.

## References and attribution
- Dataset: MaleCNS v1.0, FlyEM Project Team (HHMI Janelia), Cambridge Drosophila Connectomics Group (University of Cambridge) / MRC Laboratory of Molecular Biology, and Google Research. CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>. The data files were used unmodified; the tables and figures here are derived summaries.
- Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, et al. *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* bioRxiv, posted 9 October 2025. [doi:10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999). A peer-reviewed version is not cited because no verified journal reference was found.
- Corsi M, Urbano J. *The Treatment of Ties in Rank-Biased Overlap.* Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), Washington, DC, 14-18 July 2024. [doi:10.1145/3626772.3657700](https://doi.org/10.1145/3626772.3657700). Used only for the statement about rank-biased overlap in Section 8; page numbers not verified.
- Code, tests and documentation text: MIT Licence ([LICENSE](../LICENSE)), which does not cover the dataset.
