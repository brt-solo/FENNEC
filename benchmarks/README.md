# Structure-learning benchmark

Synthetic DAGs with d = 10 and 20 variables, n = 200 and 1000 rows, 10 seeds, and four mechanism
families (linear, additive, MLP/interaction, mixed). `generate.py` builds them; signal:noise is 4:1
by default and 1:1 with `--noise-sd 1`.

```bash
pip install -e ".[bench,experiments]"
python -m benchmarks.run                    # writes benchmarks/results/results.jsonl
python -m benchmarks.run --noise-sd 1       # writes benchmarks/results/results_noise1.jsonl
python -m benchmarks.summarize              # tables + heatmap, held-out seeds 5-9
python -m benchmarks.summarize --noise-sd 1
```

The result files used in the top-level README are included in `benchmarks/results/`, so
`summarize` works without rerunning anything.

## Methods

| name | what it is | needs |
|---|---|---|
| `fennec` | XGBoost cross-validated score, 4x BIC penalty, hill climbing | core |
| `fennec_bic`, `fennec_screened`, `linear_score` | ablations (1x BIC, screened candidate parents, linear model) | core |
| `pc`, `ges`, `gsf`, `lingam` | causal-learn | `bench` extra |
| `mmhc`, `nbsel`, `r2_sortnregress` | Python MMHC, lasso neighbourhood selection, and the R²-sortability artefact check (Reisach et al. 2023) | core + scipy |
| `notears_mlp` | NOTEARS-MLP (PyTorch implementation here) | `bench` extra |
| `grandag` | GraN-DAG via gcastle, then CAM pruning | `bench` extra, R |
| `score_sm`, `nogam` | SCORE and NoGAM, implemented here from the papers | `bench` extra, R for pruning |
| `cam`, `mmhc_bnlearn` | R packages `CAM`, `bnlearn` (`r/methods.R`) | R |
| `bidag` | R package `BiDAG` order MCMC (`r/bidag.R`) | R |

R methods call `Rscript`. If it is not on your `PATH`, set `FENNEC_RSCRIPT=/path/to/Rscript`.
The R packages needed are `CAM` (archived on CRAN; install from the CRAN archive), `mgcv`,
`bnlearn` and `BiDAG`.

## Notes on fairness

- The FENNEC penalty multiplier (4x) was chosen on seeds 0-4 and Sachs. `summarize` therefore
  reports seeds 5-9 by default; `--seeds all` pools everything.
- GraN-DAG was run with default hyperparameters and is probably under-tuned.
- `gsf` (GES with the generalised score) was too slow at n = 1000 and was run at n = 200 only.
