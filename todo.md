# Todo

## 1. Bug fix

**`evaluation/benchmarks/mmlu.py:97` — answer type mismatch**
The MMLU loader already converts the integer answer to a letter string (e.g. `"A"`), so
`["A","B","C","D"][an]` will throw a `TypeError` at runtime. The comparison should just be
`an == pd` (both strings). Verify the loader and benchmark are in sync on what type `answer`
holds before running any eval.

---

## 3. Complete DPO training

**`training/dpo.py` — finish the training loop**
The file cuts off before tokenization. Key things to work out before writing:
- DPO needs four forward passes per example: policy on chosen, policy on rejected, reference
  on chosen, reference on rejected. The collator and batch structure are different from SFT.
- The reference model is already a merged dense model (from `merge_and_unload()`). The policy
  model wraps it in a fresh LoRA — this is correct, but understand that these are new DPO
  adapters, not a continuation of the SFT adapters.
- Remeber to include test/train split

**DPO run notebook**
Mirror the structure of `01_sft_training.ipynb`.

---

## 4. MMLU rework

Three changes to make together, since they affect both the loader and the benchmark:

1. **Format matching**: the base model gets the plain prompt; SFT and SFT+DPO models get the
   same question wrapped in the Dolly instruction template. Use the actual format from
   `data/loaders/dolly.py` (`Instruction: ... \nResponse:`) — not the `### Instruction:`
   variant you may see in examples online.

2. **5-shot**: prepend five fixed exemplars to every prompt. For the base model use plain
   exemplars; for SFT/SFT+DPO wrap each exemplar in the same Dolly scaffold. Hold exemplars
   constant across all three models.

3. **Log-likelihood letter scoring**: the current implementation already does this correctly
   (logits over `" A" / " B" / " C" / " D"` at the last prompt token, argmax). Keep it.
   Just verify token IDs on a sample prompt before running the full eval.

---

## 5. lm-evaluation-harness cross-check

Write a custom MMLU task YAML that wraps `doc_to_text` in the Dolly template (for the
SFT/SFT+DPO runs) and run the harness on all three models. Target: agreement within ~1% of
your from-scratch implementation. Disagreement is the more educational outcome — it tells you
exactly where your implementation diverges.

---

## 6. IFEval

Write `evaluation/benchmarks/ifeval.py` and a run notebook. IFEval uses programmatic
constraint checks (length, keywords, format), so no judge model needed. Run on all three
models. This is the primary signal for whether SFT generalized, and the regression check for
DPO over-refusal.

---

## 7. AdvBench

Write `evaluation/benchmarks/advbench.py` and the refusal script. The config and loader
already exist. The keyword-matching refusal detector in `configs/advbench.yaml` is a
reasonable starting point for a 1B model, but be aware it will miss soft refusals and
over-count false positives (e.g. a response that quotes a keyword while complying). Run on
all three models.

---

## 8. Analysis

Compare results across models on all four metrics. The deltas between stages are the primary
signal, not the absolute scores. Specific things to look for:
- Held-out Dolly loss: did both SFT and SFT+DPO fit the target distribution?
- IFEval: did SFT improve over base? Did DPO degrade below SFT (over-refusal tax)?
- MMLU: after format-matching, does the base-vs-SFT gap shrink toward zero?
- AdvBench: did DPO raise refusal rates? By how much, relative to base and SFT?
