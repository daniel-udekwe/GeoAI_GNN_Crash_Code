# Do graph neural networks learn spatial dependence?

Code for the manuscript *"Do graph neural networks learn spatial dependence? Exposure-informed GeoAI for statewide road-crash forecasting"* (D. Udekwe and Q. Guo, FAMU-FSU College of Engineering), submitted to *Computational Urban Science*, special issue "Revisiting Spatiotemporal Modeling in the Era of GeoAI".

The pipeline forecasts quarterly crash counts on about 80,000 miles of North Carolina state-maintained roads (0.5-mile bins, 2021-2024). It compares statistical safety performance functions, gradient-boosted trees, non-graph neural networks and graph neural networks (GCN, GAT, a spatiotemporal GNN and an exposure-informed variant), and diagnoses what spatial dependence each model learns.

## Data

The study uses the North Carolina files of the FHWA **Highway Safety Information System (HSIS)**, 2021-2024: `nc21crash.csv` ... `nc24crash.csv`, `nc21road.csv` ... `nc24road.csv` (the unit files are not needed). HSIS data are available from FHWA on request (https://www.hsisinfo.org) and **cannot be redistributed**, so they are not included in this repository.

Place the files in `data/hsis_nc/`, or set the environment variable `HSIS_DIR` to their folder.

## Setup

```bash
pip install -r requirements.txt
```

Everything runs on a CPU. A 2-core machine with 7 GB of RAM runs the full study in about 8-10 hours. Run the scripts one at a time; running several in parallel can exhaust memory.

## Running the pipeline

All scripts write their outputs to the current directory, so run them from a separate working folder:

```bash
mkdir run && cd run
python ../01_build_bins.py 0.5        # fixed bins along each route, AADT/VMT, crash matching
python ../02_build_graph.py 0.5       # along-route edges + junctions inferred from crash reference roads
python ../03_prepare_features.py 0p5  # node-quarter counts, inputs, graph statistics
python ../04_exp_main.py 0p5          # all 11 models (seed ensembles)
python ../05_exp_ablations.py         # EI-STGNN ablations (offset, loss, graph, K = 1-4)
python ../06_exp_cv.py                # random vs county-blocked node cross-validation
python ../07_diagnostics.py           # learned influence by hop distance, GAT attention analysis
python ../08_bootstrap.py             # county-cluster bootstrap of skill differences
python ../09_figures.py               # figures -> figures/
python ../10_tables.py                # LaTeX table rows -> parts/
python ../11_county_metrics.py        # county- and roadway-class-level metrics
python ../12_maps_heatmap.py          # county maps and roadway-class heatmap (downloads NC county boundaries)
```

For the scale analysis, repeat steps 1-3 with `0.25` / `0p25` and `1.0` / `1p0`, then run:

```bash
python ../04_exp_main.py 0p25 x HA,NB-lag,LGBM-S,MLP,GCN,EI-STGNN
python ../04_exp_main.py 1p0  x HA,NB-lag,LGBM-S,MLP,GCN,EI-STGNN
```

## Files

| File | Purpose |
| --- | --- |
| `gm.py` | Data class, models (MLP, GRU, GCN, GAT, ST-GNN, EI-STGNN) and training loop |
| `ev.py` | Metrics: Poisson deviance skill, calibration, screening capture, residual network Moran's I and correlogram, group-wise skill |
| `01`-`03` | Data preparation and road-graph inference |
| `04`-`08` | Experiments and statistical tests |
| `09`-`12` | Figures, tables and maps |

## Protocol in brief

- **Task:** one-quarter-ahead crash counts per bin. 2021 provides lags only; training targets are 2022 Q1-2023 Q2, validation 2023 Q3-Q4 and test 2024 Q1-Q4.
- **Metric:** mean Poisson deviance, reported as skill relative to a seasonal historical average.
- **Diagnostics:** residual network Moran's I and correlogram, Jacobian-based influence distribution, GAT attention by link type, county-blocked cross-validation, and three spatial resolutions.

## Citation

If you use this code, please cite the paper (citation to be added on publication).
