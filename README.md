# Fruit Fly Connectome Explorer

How sensitive is the identification of important neurons to the connection-strength threshold used to build a fruit fly connectome graph?

**Status: Phase 1b complete (data verified and documented). The threshold analysis has not been run; there are no research results yet.**

## Research question
Rankings of neurons by outgoing connectivity are often computed after discarding weak connections. This project tests whether the choice of that cut-off changes which neurons rank highest. It is a question about a structural graph, not a claim about which neurons matter biologically.

## Data
FlyEM Male CNS connectome v1.0 (Janelia / University of Cambridge / MRC LMB / Google Research), licensed CC BY 4.0.
Source: <https://male-cns.janelia.org/download/>. The data are not stored in this repository.

Note: connection endpoints are segmentation *body* IDs, not neuron IDs. Of 88.4 M bodies with synapses, only about 0.2 M have annotations, and neurons are identified through those annotations (see the data dictionary).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Further documentation: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## License
Code license to be chosen in a later phase. The dataset's CC BY 4.0 terms apply to the data only.
