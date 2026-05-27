# refusal-representations

## Project overview
A study of safety finetuning in LLMs. Specifically, we finetune Llama 3.2 1B 
(base) on a subset of Anthropic's HH-RLHF dataset using DPO + LoRA, focused 
on the refusal of harmful requests. We then analyze how the finetuning changes
the model's internal representations.

The project has three components, built sequentially:
1. Finetune + evaluate (measure behavioral shift via refusal rate on AdvBench)
2. Linear probes (locate where safety-relevant information becomes linearly 
   represented across layers)
3. Refusal direction analysis (extract via difference-in-means, test causal 
   role via ablation/addition)

Currently in progress: Component 1, finetuning and evaluation.

## Architecture principles
- **Abstract interfaces, swappable implementations.** `base.py` files define 
  what a data loader or benchmark must do. Concrete files implement that 
  contract. Swapping AdvBench for HarmBench means changing a config, not 
  touching training or eval logic.
- **Config-driven.** Hyperparameters live in `configs/`, not hardcoded in 
  source files. 
- **Notebooks are thin.** All real logic lives in importable Python modules. 
  Notebooks just orchestrate and visualize.

## Directory structure
data/loaders/      # base.py defines interface; hh_rlhf.py, advbench.py implement it
models/            # single entry point: loader.py
training/          # dpo.py — DPO + LoRA finetuning logic
evaluation/        # refusal.py measures refusal rate; benchmarks/ are swappable
interpretability/  # probing/ and directions/ — built in components 2 and 3
utils/             # logging, checkpoint helpers
configs/           # YAML configs for finetuning and eval
notebooks/         # one notebook per component
scripts/           # CLI entry points wrapping the modules

## Conventions
- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public functions and classes
- Config loaded via YAML, passed as dicts or dataclasses — never os.environ 
  in business logic
- Never hardcode model names, paths, or hyperparameters in source files
- Secrets (HF token, W&B key) via environment variables only, never committed

## Key dependencies
- `transformers`, `peft` — model loading and LoRA
- `trl` — DPO training
- `datasets` — HH-RLHF and benchmark loading
- `torch` — everything
- `wandb` — experiment tracking
- `TransformerLens` — interpretability work (components 2 and 3)

## Compute context
Primary: Google Colab Pro (GPU). Code must be importable from a cloned repo 
inside Colab. Heavy outputs (checkpoints, activations) save to Google Drive. 
Do not assume a persistent local filesystem.

## What not to do
- Do not modify files in `interpretability/` yet
- Do not hardcode any file paths; use `pathlib.Path` and config values
- Do not add dependencies outside the key dependencies above without flagging it