# Figures and captions

Derived from **MaleCNS v1.0** (FlyEM Project Team at HHMI Janelia, Cambridge Drosophila Connectomics Group / MRC LMB, Google Research), licensed CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>); source <https://male-cns.janelia.org/download/>; preprint doi:10.1101/2025.10.09.680999. The data files were used unmodified; these figures are derived summaries. The MIT licence of this repository does not cover the data.

## Definitions used in every figure

- **Threshold T:** keep a connection record when its weight is at least T (≥). Thresholds: 1, 2, 5, 10, 20, 50. Weight = number of synaptic connection records between a pair of bodies, per the data providers’ code (the dataset documentation does not define it in prose).
- **Primary population:** bodies with a `superclass` annotation (166,700). Provisional: it rests on the annotation semantics given in the preprint. The population is fixed across thresholds, including neurons left with no connection.
- **Fully traced subset (sensitivity analysis):** `superclass` present **and** `status == "Traced"` (164,606 bodies). It is a sensitivity analysis, not a superior biological definition; it excludes many sensory bodies labelled “Out of scope” (1,982 `ol_sensory` bodies among the 2,094 it drops). Its top lists are identical or almost identical to the primary population’s, so the figures show the primary population.
- **Edge categories (source → target):** N = a body of the primary population; **O = “other body” = every body not in that population** (annotated bodies without a superclass and all unannotated bodies). NN, NO, ON, OO.
- **Metrics:** out-strength = summed weight of outgoing records to neuron partners; out-degree = number of distinct downstream neuron partners (each record is a distinct pair; no duplicate pairs exist).
- **Ties:** headline top-K sets break ties by body ID (exactly K neurons); the tie-inclusive view keeps every neuron tied at the K-th value. Both are always shown; boundary ties are flagged.
- The 166,691 (preprint) versus 166,700 (this primary population) count difference is unresolved and is not adjusted for. Findings apply to this single dataset version and these graph definitions. Rank is a structural measure, not evidence of biological importance.

## Figure 1 – weight distribution (`fig1_weight_distribution.png`)

Number of body pairs at each connection weight (log–log), with the six thresholds. Shares of records kept: T≥1: 100.00%, T≥2: 37.98%, T≥5: 5.02%, T≥10: 1.84%, T≥20: 0.70%, T≥50: 0.15%. Computed from `threshold_totals.csv` and checked against the weight histogram. Descriptive; no functional form is claimed.

## Figure 2 – what survives (`fig2_retention.png`)

(a) Share of records and of summed weight kept (log scale): at T≥50, 0.15% of records but 6.6% of the weight. (b) Share of the fixed 166,700 primary neurons with an outgoing, incoming, any, or no neuron-to-neuron connection: any = 100%, 100%, 99%, 98%, 87%, 40% from T≥1 to T≥50; none = 0%, 0%, 1%, 2%, 13%, 60%.

## Figure 3 – edge-category composition (`fig3_edge_categories.png`)

Share of retained records and of summed weight in each category (primary population). Neuron-to-neuron share of records: T≥1: 17%, T≥2: 27%, T≥5: 82%, T≥10: 98%, T≥20: 100%, T≥50: 100%; of weight: T≥1: 40%, T≥2: 52%, T≥5: 92%, T≥10: 99%, T≥20: 100%, T≥50: 100%. Raising the threshold therefore also changes the composition of the graph.

## Figure 4 – top-K overlap (`fig4_topk_overlap.png`)

Jaccard similarity of the top 10 / 50 / 100 neurons at each threshold against the T≥1 list (top row) and the T≥10 list (bottom row), for out-strength (left) and out-degree (right); filled markers and lines are the headline view, “+” the tie-inclusive view, open markers a tie at the K-th place. Against T≥1, top-10 Jaccard at T≥10 is 0.67 for strength and 0.11 for degree; top-100: 0.64 and 0.17. Boundary ties: 0 of 18 strength rows and 3 of 18 degree rows (at most 8 neurons share the K-th value).

## Figure 5 – all-pairs stability (`fig5_pairwise_stability.png`)

Jaccard (top 50, headline view) and Spearman rank correlation over all 166,700 neurons for every pair of thresholds. Spearman caution: neurons with no outgoing neuron-to-neuron connection tie at value 0 — 117,564 neurons (71%) at T≥50 versus 1,220 (0.7%) at T≥1 — so correlations involving high thresholds partly reflect which neurons keep any connection.

## Figure 6 – rank trajectories (`fig6_rank_trajectories.png`)

Rank of every neuron that is in the top 10 at T≥1, T≥10 or T≥50, followed across all six thresholds (rows: strength, degree; columns: the threshold that selected them; a neuron selected at several thresholds appears in each). Neurons with no surviving outgoing neuron connection are marked × and given no rank. The underlying table, with selection source, type, superclass, rank and survival flag per neuron and threshold, is `outputs/tables/rank_trajectories_<metric>.csv`.

## Figure 7 – value distributions (`fig7_value_distributions.png`)

Number of primary neurons with at least a given neuron-to-neuron out-strength (left) or out-degree (right), one curve per threshold (positive values only). The value of the 100th-ranked neuron (where each curve crosses 100 neurons) is, for T≥1 / T≥10 / T≥50: out-strength 18,662 / 12,831 / 6,600; out-degree 3,419 / 447 / 60. Between T≥1 and T≥50 the 100th-ranked value falls 2.8-fold for out-strength but 57-fold for out-degree, so the upper tail of out-strength moves comparatively little with the threshold while the out-degree distribution shifts a great deal, which is consistent with (but does not by itself prove) the lower stability of degree-based lists in Figure 4. Descriptive; no functional form is claimed.

Palette: light mode, validated with a colour-palette checker (lightness band, chroma floor, adjacent-pair colour-blind separation, normal-vision floor, contrast on the chart surface all pass); identity is also carried by markers and labels.
