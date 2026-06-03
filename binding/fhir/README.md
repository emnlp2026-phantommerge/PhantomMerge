# FHIR-AgentBench integration

Upstream benchmark: **https://github.com/glee4810/FHIR-AgentBench**

## Rollouts

Follow upstream setup (conda env, MIMIC-IV-FHIR, `run_agent.py`, vLLM). Use the agent output JSON path produced by your run.

## Phantom-Merge judge

From the PhantomMerge repo root:

```bash
python binding/fhir/scripts/run_pm_judge.py \
  --rollout /path/to/agent_output.json \
  --out-dir runs/fhir_pm_eval_out \
  --judge-base-url http://127.0.0.1:8001/v1
```

Optional `--eval-json` for task-level eval fields; `--rivals-csv` for PM-subset lists from upstream. Protocol: `fhir_pm_judge_vNEXT`.

Sealed paper labels: `results/table1_characterization/fhir_qwen_n973/` and `fhir_mistral_n847/`.
