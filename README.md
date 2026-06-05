# llama-sft-dpo-lora
Implementing a full post-training pipeline on Llama 3.2 1B from scratch:
- SFT on 5k examples from databricks-dolly-15k for general assistant response behavior.
- DPO on 5k examples from hh-rlhf harmless-base for refusal of harmful user requests.
- Safety evaluation on the AdvBench benchmark.
- Capability evaluation on 1k examples from the MMLU benchmark.