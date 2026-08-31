"""
Step 6 verification for the MMLU rework. CPU-only.

Checks:
  1. Loader emits the expected raw fields and a per-subject few-shot pool.
  2. Assembled prompts (both styles) end on the answer cue with no trailing space.
  3. Appending " A".." D" to a real assembled prompt tokenizes to the exact
     single answer-letter token (no merge with ':', no double space).
  4. Assembled prompt token lengths vs the configured max_length.

Run:  PYTHONPATH=. python tests/prompttest.py
"""

import yaml
from transformers import AutoTokenizer

from evaluation.benchmarks.mmlu import MMLUBenchmark

MODEL_NAME = "meta-llama/Llama-3.2-1B"
STYLES = ["plain", "dolly"]
N_LEN_SAMPLES = 100


def main():
    with open("./configs/mmlu.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # load() builds data + fewshot + prompt_style from the config
    # (per-subject filtering over test + dev takes a minute or two)
    bench = MMLUBenchmark(config, None, tokenizer)
    bench.load()

    # ---- 1. loader output ----
    print("=" * 72)
    print("LOADER OUTPUT")
    print("=" * 72)
    ex = bench.data[0]
    print("data[0] keys :", list(ex.keys()))
    print("  subject    :", ex["subject"])
    print("  question   :", ex["question"][:120])
    print("  choices    :", ex["choices"])
    print("  answer_letter:", repr(ex["answer_letter"]),
          "(type", type(ex["answer_letter"]).__name__ + ")")
    print("fewshot subjects :", len(bench.fewshot))
    subj_shots = bench.fewshot[ex["subject"]]
    print(f"fewshot['{ex['subject']}'] count :", len(subj_shots))
    print("  exemplar[0] keys   :", list(subj_shots[0].keys()))
    print("  exemplar[0] answer_letter :", repr(subj_shots[0]["answer_letter"]))

    # answer-letter token ids (must each be a single token)
    letter_ids = {}
    for L in ["A", "B", "C", "D"]:
        enc = tokenizer.encode(" " + L, add_special_tokens=False)
        letter_ids[L] = enc[-1]
        tag = "OK single token" if len(enc) == 1 else "!! MULTI TOKEN"
        print(f'  encode(" {L}") -> {enc}  {tag}')
        assert len(enc) == 1, f'" {L}" is not a single token'

    # ---- 2 + 3. per-style prompt + token checks ----
    for style in STYLES:
        bench.prompt_style = style
        prompt = bench._build_prompt(bench.data[0])

        print("\n" + "=" * 72)
        print(f"STYLE: {style}")
        print("=" * 72)
        print(prompt)
        print("-" * 72)
        print("last 20 chars:", repr(prompt[-20:]))
        assert not prompt[-1].isspace(), f"[{style}] prompt ends with whitespace"

        for L in ["A", "B", "C", "D"]:
            ids = tokenizer.encode(prompt + " " + L, add_special_tokens=False)
            last, want = ids[-1], letter_ids[L]
            status = "OK" if last == want else "!! MISMATCH"
            print(f'  prompt + " {L}"  last id = {last:6d}  (want {want:6d})  {status}')
            assert last == want, f"[{style}] ' {L}' does not tokenize cleanly at prompt end"

    # ---- 4. length check ----
    max_len = config["evaluation"]["max_length"]
    n = min(N_LEN_SAMPLES, len(bench.data))
    print("\n" + "=" * 72)
    print(f"PROMPT LENGTHS vs max_length={max_len}  (first {n} examples)")
    print("=" * 72)
    for style in STYLES:
        bench.prompt_style = style
        lengths = [
            len(tokenizer(bench._build_prompt(bench.data[i]))["input_ids"])
            for i in range(n)
        ]
        over = sum(x > max_len for x in lengths)
        print(f"  {style:5s}  min={min(lengths):4d}  mean={sum(lengths) // len(lengths):4d}  "
              f"max={max(lengths):4d}  over_max={over}/{n}")

    print("\nall assertions passed.")


if __name__ == "__main__":
    main()
