from data.loaders.dolly import DollyDataLoader
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch


def tokenize_prompt_response(example: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenizes a (prompt, response) pair and masks the prompt tokens so only
    the response contributes to loss. Shared by sft.py (its own training
    data) and dpo.py (chosen/rejected are each a prompt/response pair, and
    the held-out Dolly degradation check).
    """
    tokenized_prompt = tokenizer(example["prompt"], padding=False)
    tokenized_response = tokenizer(example["response"], padding=False,
                                   add_special_tokens=False)

    prompt_ids = tokenized_prompt["input_ids"]
    response_ids = tokenized_response["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + response_ids)[:max_length]

    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def split_dolly(sft_config: dict):
    """
    Loads Dolly and applies the canonical 95/5 train/test split, using
    sft_config's own data/seed settings. Always called with configs/sft.yaml
    (even from dpo.py) so the held-out 5% is guaranteed to be the exact same
    rows sft.py trained against, not just a same-seed coincidence.
    """
    dataloader = DollyDataLoader(sft_config)
    dataloader.load()
    dataset = dataloader.get_data()

    split = dataset.train_test_split(test_size=0.05, seed=sft_config["data"]["seed"])
    return split["train"], split["test"]


def evaluate_holdout_loss(model, tokenizer, holdout_data, device,
                          max_length: int, batch_size: int) -> float:
    """
    Held-out Dolly cross-entropy loss: the SFT training-success gate, and
    (called on the DPO policy model) the check for whether preference
    training degraded the Dolly fit. Token-weighted average over the split.

    Sets the model to eval mode; the caller is responsible for switching
    it back to train() afterward.
    """
    tokenized = holdout_data.map(
        tokenize_prompt_response,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=holdout_data.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True,
                                      label_pad_token_id=-100)
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False,
                        collate_fn=collator)

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            n_tokens = (batch["labels"] != -100).sum()
            total_loss += outputs.loss.float() * n_tokens
            total_tokens += n_tokens

    return (total_loss / total_tokens).item()
