# Todo

### SFT
somethings still wrong i think, need to check.

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
