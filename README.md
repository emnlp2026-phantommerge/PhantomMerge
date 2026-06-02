# Phantom Merge (EMNLP 2026 — anonymous release)

Code and sealed metrics for [*Phantom Merge: When Your Large Language Model Agents Pick One but Tell You About Another*](paper/Phantom_Merge__emnlp2026_.pdf).

**Phantom Merge** is a binding failure in multi-step agents: the run commits to a protocol anchor (a ShoppingBench product or FHIR resource) but the final answer attributes to that anchor facts its observed evidence cannot support. Standard task checks may still pass.

This repository ships **precomputed trajectory labels and paper tables** for four cohorts (ShoppingBench and FHIR-AgentBench × Qwen3-32B / Mistral-Small-3.1-24B). It does **not** include agent rollouts, judge API endpoints, or feature-bank hidden-state arrays.

**Repository:** [https://github.com/emnlp2026-phantommerge/PhantomMerge](https://github.com/emnlp2026-phantommerge/PhantomMerge)

## Quick check (CPU, ~2 min)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce/verify.sh
```

Expected output includes `Release check PASSED`. Paper headline numbers:

```bash
python scripts/print_paper_metrics.py
```

`reproduce/verify.sh` checks row counts and SHA-256 on the four `per_trajectory.jsonl` files, validates probe manifests (`claims.parquet`, feature-bank index), and compares aggregates to `results/table1_characterization/counts.json`. It does **not** call an LLM judge.

## Paper numbers → files

| Paper | Location |
|-------|----------|
| Table 1 — PM prevalence (N = 249 / 250 / 973 / 847) | `results/table1_characterization/counts.json`, `*/per_trajectory.jsonl` |
| Table 2 — global-support baseline | `results/table2_global_support/baseline_checker.json` |
| Table 3 — BCP-Detect (FHIR Qwen test AUROC 0.904) | `results/table3_representation/bcp_detect.json`, `claims.parquet` |
| Table 4 — MSPS (test PM 35.6% → 4.8%) | `results/table4_mitigation/msps_test146.json` |
| Appendix — τ curves, CB retention audit | `results/appendix/` |

Machine-readable index: [`ARTIFACT.json`](ARTIFACT.json).

## Repository layout

```
binding/shopping/   PM judge + object KB builder (ShoppingBench)
binding/fhir/       PM judge (FHIR-AgentBench); shared claim taxonomy
probes/             BCP / OC-BCP / MSPS training and evaluation code
results/            Sealed labels and tables (authoritative for the paper)
reproduce/          verify.sh and optional re-run scripts
figures/appendix/   Regenerate appendix CSVs/plots (CPU)
```

## Optional re-runs (GPU; not needed to verify the paper)

Sealed files under `results/` are the reference for published numbers.

```bash
bash scripts/run_diagnosis.sh    # Table 1 integrity only (still no judge)
bash scripts/run_probe.sh        # BCP-Detect pipeline (needs regenerated .npy)
bash scripts/run_mitigation.sh   # MSPS / gating suite
```

Re-labeling trajectories with `binding/*/scripts/analyze_phantom_merge_*.py` requires the original benchmark rollouts, which are not redistributed here.

## Label protocol

Rows in `per_trajectory.jsonl` carry `pipeline_version`: `shopping_pm_judge_vNEXT` or `fhir_pm_judge_vNEXT`. Claim taxonomy and PM aggregation logic live in `binding/shopping/src/analysis/pm_eval_vnext.py`.

## Not redistributed

- ShoppingBench / FHIR-AgentBench rollouts and upstream benchmark trees
- Feature-bank tensors (`*.npy`)
- Human label-audit bundles or qualitative case trajectories
- 22 FHIR-Mistral trajectories dropped from the sealed cohort (count only in `results/table1_characterization/cohort_index.json`)

## Citation

```bibtex
@inproceedings{phantommerge2026,
  title     = {Phantom Merge: When Your Large Language Model Agents Pick One but Tell You About Another},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year      = {2026},
}
```

See also [`CITATION.bib`](CITATION.bib).

## License

MIT ([`LICENSE`](LICENSE)).
