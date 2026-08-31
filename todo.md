# Todo

## Next steps

### 1. Gate: is the SFT checkpoint current?
A base-vs-SFT comparison only means something if the SFT adapter is properly trained. The
notes say SFT loss was flat at LR 2e-5; commit `55e670c` bumped it to 2e-4.
- If the checkpoint on Drive is from the **2e-4** run with the current script -> proceed.
- If it's the old **2e-5** flat run -> **retrain SFT first** with the updated `sft.py`
  (picks up the val-loss curve, EOS, grad clip). On the critical path anyway.

### 2. MMLU — small run (Step 7)
Trim `subjects` in `configs/mmlu.yaml` to ~4, run `capabilities.py` for base + SFT. Confirm
(a) no crash, (b) the ~2.5% base-over-SFT gap shrinks toward zero or flips slightly once
format-matched. If a real gap survives, that's the reportable finding. Restore the full
subject list after.

### 3. MMLU — full run + harness cross-check (Step 8)
Full 57-subject run for every model that exists (base + SFT now; DPO row after DPO training).
Then a custom lm-evaluation-harness MMLU task YAML with a `doc_to_text` that reproduces the
Dolly wrapper for the SFT/DPO runs. Target: agreement within ~1%. Disagreement localizes an
implementation gap and is the more educational outcome.

### 4. DPO training
**`training/dpo.py`** cuts off before tokenization. Before writing:
- Four forward passes per example: policy on chosen, policy on rejected, reference on chosen,
  reference on rejected. Collator / batch structure differ from SFT.
- Reference model is already a merged dense model (`merge_and_unload()`); policy wraps it in a
  **fresh** LoRA — new DPO adapters, not a continuation of the SFT adapters.
- Loss: `-log sigmoid(beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected)))`.
  Add `beta` to `configs/dpo.yaml` (missing).
- Same held-out Dolly val-loss hook as `sft.py` — factor the eval loop into a shared helper
  so both call it. Measures whether preference training degraded the Dolly fit.
- Train/test split, mirroring `sft.py`.

**DPO run notebook** — mirror `notebooks/01_sft_training.ipynb`.

Then: held-out Dolly loss gate for SFT+DPO (should be close to SFT's).

### 5. IFEval
Write `evaluation/benchmarks/ifeval.py` + a run notebook. Programmatic constraint checks
(length, keywords, format) — no judge model. Run on all three models. Primary signal for
whether SFT instruction-following *generalized* beyond Dolly; doubles as the DPO
over-refusal regression check (SFT+DPO dropping materially below SFT = the helpfulness tax).

### 6. AdvBench
Write `evaluation/benchmarks/advbench.py` + the refusal script. Config and loader exist. The
keyword refusal detector in `configs/advbench.yaml` is a reasonable start for a 1B model but
will miss soft refusals and over-count false positives (response quoting a keyword while
complying). Run on all three models.

### 7. Analysis
Compare deltas between stages (primary signal, not absolute scores):
- Held-out Dolly loss: did SFT and SFT+DPO both fit the target distribution?
- IFEval: did SFT improve over base? Did DPO degrade below SFT (over-refusal tax)?
- MMLU: after format-matching, did the base-vs-SFT gap shrink toward zero?
- AdvBench: did DPO raise refusal rates, and by how much vs base and SFT?
