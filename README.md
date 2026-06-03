# Phantom Merge (EMNLP 2026 — anonymous release)

Code and reference runs for [*Phantom Merge: When Your Large Language Model Agents Pick One but Tell You About Another*](https://github.com/emnlp2026-phantommerge/PhantomMerge).

**Phantom Merge** is a binding failure in multi-step agents: the run commits to a protocol anchor (a ShoppingBench product or a FHIR resource) but the final answer attributes to that anchor facts its observed evidence cannot support. Standard task checks may still pass.

This artifact implements Phantom-Merge **diagnosis** (anchor–claim labeling) and **mitigation** (BCP-Detect, MSPS) on two agent benchmarks—[ShoppingBench](https://github.com/yjwjy/ShoppingBench) and [FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench)—with **Qwen3-32B** and **Mistral-Small-3.1-24B** backbones. Sealed trajectories and paper tables are under `results/`; `binding/` and `probes/` are the code paths to regenerate from fresh rollouts.

## Installation

```bash
git clone https://github.com/emnlp2026-phantommerge/PhantomMerge.git
cd PhantomMerge
git lfs pull

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce/verify.sh
```

Prints `Release check PASSED` when the sealed snapshot matches the paper. Headline metrics:

```bash
python scripts/print_paper_metrics.py
```

Index: [`ARTIFACT.json`](ARTIFACT.json).

## Agent benchmarks

Rollouts are produced in the upstream repos; this repo consumes rollout logs and runs the PM judge and downstream probes.

### ShoppingBench

- **Upstream:** https://github.com/yjwjy/ShoppingBench  
- **Paper:** https://arxiv.org/abs/2508.04266  
- **Cohorts in this release:** `shopping_qwen_n249`, `shopping_mistral_n250`

1. Clone ShoppingBench and follow its environment setup (`./init_env.sh`, product sandbox, API keys as in upstream `README.md`).
2. Run agent rollouts, e.g. `./run.sh product rollout <model>` (see upstream for shop / voucher / web intents).
3. Point the PM judge at your rollout JSONL:

```bash
python binding/shopping/scripts/run_pm_judge.py \
  --rollout /path/to/shoppingbench/rollout.jsonl \
  --out-dir runs/shopping_pm_eval_out \
  --judge-enabled
```

4. Compare against sealed labels: `results/table1_characterization/shopping_*/per_trajectory.jsonl`, or run `bash reproduce/table1_characterization.sh`.

Details: [`binding/shopping/README.md`](binding/shopping/README.md).

### FHIR-AgentBench

- **Upstream:** https://github.com/glee4810/FHIR-AgentBench  
- **Paper:** https://arxiv.org/abs/2509.19319  
- **Cohorts in this release:** `fhir_qwen_n973`, `fhir_mistral_n847`

1. Clone FHIR-AgentBench; create the conda env and install `requirements.txt` per upstream.
2. Prepare MIMIC-IV-FHIR and run agents with `run_agent.py` (vLLM / tool-use setup as in upstream).
3. Run the PM judge on agent output JSON:

```bash
python binding/fhir/scripts/run_pm_judge.py \
  --rollout /path/to/fhir_agent_output.json \
  --out-dir runs/fhir_pm_eval_out
```

4. Sealed reference: `results/table1_characterization/fhir_*/per_trajectory.jsonl`; Table 1 check: `bash reproduce/table1_characterization.sh`.

Details: [`binding/fhir/README.md`](binding/fhir/README.md).

## Reproducing the paper

| Paper | Location |
|-------|----------|
| Table 1 — PM prevalence | `results/table1_characterization/counts.json`, `*/per_trajectory.jsonl` |
| Table 2 — global-support baseline | `results/table2_global_support/baseline_checker.json` |
| Table 3 — BCP-Detect | `results/table3_representation/bcp_detect.json`, `claims.parquet` |
| Table 4 — MSPS | `results/table4_mitigation/msps_test146.json` |
| Appendix | `results/appendix/` |

End-to-end scripts (GPU where noted):

```bash
bash reproduce/table3_bcp_detect.sh   # Table 3
bash reproduce/table3_oc_bcp.sh
bash reproduce/table4_mitigation.sh   # Table 4
```

Shortcut wrappers: `scripts/run_probe.sh`, `scripts/run_mitigation.sh`. Feature-bank training for BCP-Detect lives under `probes/`; build tensors from your probe run before re-invoking Table 3 scripts.

## Code layout

```
binding/shopping/   PM judge + object KB (ShoppingBench)
binding/fhir/       PM judge (FHIR-AgentBench)
probes/             BCP-Detect, OC-BCP, MSPS
results/            Sealed per-trajectory labels and tables
reproduce/          verify.sh and table-aligned scripts
figures/appendix/   Appendix figures (CPU)
```

## Label protocol

`per_trajectory.jsonl` rows use `pipeline_version` `shopping_pm_judge_vNEXT` or `fhir_pm_judge_vNEXT`. Shared claim taxonomy and aggregation: `binding/shopping/src/analysis/pm_eval_vnext.py`.

## Citation

```bibtex
@inproceedings{phantommerge2026,
  title     = {Phantom Merge: When Your Large Language Model Agents Pick One but Tell You About Another},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year      = {2026},
}
```

See [`CITATION.bib`](CITATION.bib).

## License

MIT ([`LICENSE`](LICENSE)).
