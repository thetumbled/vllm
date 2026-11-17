# Last-layer FFN pruning: validation evidence

This branch contains benchmark and regression evidence for the feature branch
[`codex/last-layer-ffn-pruning`](https://github.com/thetumbled/vllm/tree/codex/last-layer-ffn-pruning).
It is separate from the implementation so raw experiment artifacts do not enter
the upstream source tree.

## Revisions and environment

- Feature: `0a14d0ddf132388555e8e17e5d5866598ed4f541`.
- Rebased main: `c84273be384d7faaeadfb36caced2f9d5a112f7f`.
- GPU: NVIDIA RTX 4090, 24 GB; BF16; FlashAttention 2.
- Current environment and measured revision: `pr-ready-validation/environment.json`.
- The editable build's package version still contains the old build SHA. The
  Python checkout is the feature revision above; there were no C++/CUDA or
  dependency changes in the 26 upstream commits between the two baselines, so
  the compiled extensions were reused.

## Offline generation comparison

`qwen25-offline-summary.json` summarizes the previous two-round ABBA experiment.
`qwen25-offline-evidence.tar.gz` contains the exact source snapshot, patch, full
scripts, commands, raw per-run timings, output/logprob comparisons and logs.
Its upstream base is `2440ffc3870cdd6329bdf7ec0f780e05a2f01eb0`, as recorded in
`manifest.json`.

Each case uses 2 warmups and 7 timed `LLM.generate` calls, output length 1,
with breakable prefill graphs enabled in both arms. These are offline generation
latencies, not HTTP request latencies or isolated kernel timings. Model load,
graph capture and warmup are excluded from these timings.

## Current-revision validation

`pr-ready-validation.tar.gz` contains the complete reproducible harness,
server startup commands, evaluator/benchmark commands, JSON results, JUnit XML,
and logs. Successful commands are in `commands.jsonl`; retained `attempt-*`
directories document setup corrections, not changes to model code or scoring.

- Regressions: 37 passed with `VLLM_GPU_SYNC_CHECK=error`.
- Accuracy: the repository's `tests/evals/gsm8k/gsm8k_eval.py`, first 128 GSM8K
  test questions, 5-shot chat prompts, temperature 0, seed 42, max output 1024.
  Requests are serialized (`--max-concurrency 1`) so every prompt is a complete
  prefill rather than a mixed prefill/decode batch that bypasses pruning.
- Serving: official `vllm bench serve`, fixed input lengths 2048/4096,
  concurrency 1/4 respectively, output length 1, temperature 0, seed 42,
  4 warmup requests and 32 measured requests per run, two runs per case and arm.
  `--random-range-ratio 0` selects fixed lengths on this revision.
- Both arms use the same feature checkout; stock disables the opt-in flag and
  pruned enables it. Both enable breakable prefill graphs and disable prefix
  caching and torch compilation. Server order is stock then pruned; the two
  HTTP repeats are a small serving check, not independent alternating runs.

The raw dataset is not duplicated here. `dataset-manifest.json` records hashes
and row counts of the existing original GSM8K train/test JSONL files. The
repository evaluator can download these datasets through its standard loader.

## Reproduce

Extract the current validation archive. The exact server commands and environment
switches are in `stock-server-command.json` and `pruned-server-command.json`.
All evaluator and benchmark commands are in `commands.jsonl`. Replace the local
model/repository/output paths for your machine. `run.py` is the restartable
controller used for this run; it preserves completed stages.

Regression command, run from the repository root with local model redirection
(or the corresponding Hub checkpoints available):

```bash
VLLM_GPU_SYNC_CHECK=error \
VLLM_MODEL_REDIRECT_PATH=/path/to/model-redirect.json \
.venv/bin/python -m pytest -v -s \
  tests/v1/cudagraph/test_last_layer_ffn.py \
  tests/v1/cudagraph/test_breakable_cudagraph.py \
  --junitxml=regressions.xml
```

`pre-commit.log` records the full successful run of:

```bash
.venv/bin/pre-commit run --from-ref origin/main --to-ref HEAD
```

## Results

- GSM8K: stock 113/128 (88.28%), pruned 113/128 (88.28%), no invalid responses.
- HTTP serving: see [summary.json](pr-ready-validation/summary.json) and individual run JSONs for timings and variability.
