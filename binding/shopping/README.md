# ShoppingBench integration

Upstream benchmark: **https://github.com/yjwjy/ShoppingBench**

## Rollouts

Install and run agents per the upstream repo (`./init_env.sh`, `./run.sh <intent> rollout <model>`). Export or locate the rollout JSONL your backbone produced.

## Phantom-Merge judge

From the PhantomMerge repo root:

```bash
python binding/shopping/scripts/run_pm_judge.py \
  --rollout /path/to/rollout.jsonl \
  --out-dir runs/shopping_pm_eval_out \
  --judge-enabled \
  --judge-base-url http://127.0.0.1:8001/v1
```

Optional `--synthesize` for ShoppingBench metadata JSONL. Protocol: `shopping_pm_judge_vNEXT` (`binding/shopping/src/analysis/pm_eval_vnext.py`).

Sealed paper labels: `results/table1_characterization/shopping_qwen_n249/` and `shopping_mistral_n250/`.
