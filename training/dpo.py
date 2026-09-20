import yaml
from data.loaders.hh_rlhf import HHRLHFDataLoader
from transformers import get_linear_schedule_with_warmup
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

from training.validation import split_dolly, tokenize_prompt_response, evaluate_holdout_loss

# load configs -- dpo.yaml for this run, sft.yaml so we can rebuild the exact
# held-out Dolly split sft.py trained against
with open("./configs/dpo.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("./configs/sft.yaml", "r") as file:
    sft_config = yaml.safe_load(file)

# load + hold out the preference data (mirrors sft.py's split, but here the
# held-out slice is used for a held-out DPO loss/accuracy check)
dataloader = HHRLHFDataLoader(config)
dataloader.load()
dataset = dataloader.get_data()
split_dataset = dataset.train_test_split(test_size=0.05, seed=config["data"]["seed"])
train_data = split_dataset["train"]
test_data = split_dataset["test"]

# the Dolly split sft.py actually trained against, to check whether DPO
# training degrades the Dolly fit
_, dolly_holdout = split_dolly(sft_config)

# load reference model and tokenizer
reference_model, tokenizer = load_model(config, checkpoint_path=
                                        config["model"]["checkpoint_path"])
reference_model.eval()                  # set to eval mode to disable dropout
reference_model.requires_grad_(False)   # explicit freeze / stop gradient flow

# load another instantiation of SFT model as the base for DPO training
base_model, _ = load_model(config, checkpoint_path=
                           config["model"]["checkpoint_path"])

# construct the lora / policy model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
)
policy_model = get_peft_model(base_model, lora_config)

# right-padding: dpo_collate below assumes real tokens come first, padding
# after -- required for the loss_mask construction to line up correctly
tokenizer.padding_side = "right"


def tokenize_pair(example: dict) -> dict:
    """
    Tokenizes (prompt, chosen) and (prompt, rejected) as two independent
    prompt/response pairs via the same helper sft.py uses, then keeps just
    the input_ids and a boolean loss_mask (response-token positions) for
    each side.
    """
    chosen = tokenize_prompt_response(
        {"prompt": example["prompt"], "response": example["chosen"]},
        tokenizer, config["model"]["max_length"])
    rejected = tokenize_prompt_response(
        {"prompt": example["prompt"], "response": example["rejected"]},
        tokenizer, config["model"]["max_length"])

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_loss_mask": [label != -100 for label in chosen["labels"]],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_loss_mask": [label != -100 for label in rejected["labels"]],
    }


train_tokenized = train_data.map(tokenize_pair, remove_columns=train_data.column_names)
test_tokenized = test_data.map(tokenize_pair, remove_columns=test_data.column_names)


def dpo_collate(batch: list) -> dict:
    """
    Pads a batch of chosen/rejected pairs into one concatenated [2*B, L]
    tensor -- rows [0:B] are chosen, rows [B:2B] are rejected -- so a single
    forward pass per model covers both sides instead of four separate calls.
    """
    chosen = [{"input_ids": ex["chosen_input_ids"]} for ex in batch]
    rejected = [{"input_ids": ex["rejected_input_ids"]} for ex in batch]
    padded = tokenizer.pad(chosen + rejected, return_tensors="pt")

    loss_masks = ([ex["chosen_loss_mask"] for ex in batch] +
                  [ex["rejected_loss_mask"] for ex in batch])
    max_len = padded["input_ids"].shape[1]
    loss_mask = torch.zeros((len(loss_masks), max_len), dtype=torch.bool)
    for i, m in enumerate(loss_masks):
        loss_mask[i, :len(m)] = torch.tensor(m, dtype=torch.bool)

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "loss_mask": loss_mask,
    }


train_dataloader = DataLoader(train_tokenized,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=dpo_collate)

test_dataloader = DataLoader(test_tokenized,
                             batch_size=config["training"]["batch_size"],
                             shuffle=False, collate_fn=dpo_collate)


def sequence_logps(logits: torch.Tensor, input_ids: torch.Tensor,
                   loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Sum of log P(token_t | prefix) under the model, over just the response
    span marked by loss_mask. logits[:, t] predicts input_ids[:, t+1], so
    everything shifts by one before gathering.
    """
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=shift_ids.unsqueeze(-1)).squeeze(-1)

    return (token_logps * shift_mask).sum(dim=-1)


beta = config["training"]["beta"]

# optimizer: AdamW -- policy only, reference is frozen
optimizer = torch.optim.AdamW(policy_model.parameters(),
                              lr=config["training"]["learning_rate"])

total_steps = ((len(train_dataloader) //
               config["training"]["gradient_accumulation_steps"]) *
               config["training"]["num_epochs"])

# lr scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config["training"]["warmup_steps"],
    num_training_steps=total_steps)

# train mode
policy_model.train()

# training-relevant initializations
acc_loss = 0
step = 1            # how many forward passes are performed
global_step = 0     # how many actual optimizer steps taken

total_epochs = config["training"]["num_epochs"]
gradient_accumulations = config["training"]["gradient_accumulation_steps"]

wandb.init(project=config["wandb"]["project"],
           name=config["wandb"]["run_name"],
           config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_model = policy_model.to(device)
reference_model = reference_model.to(device)

# just in case
optimizer.zero_grad()

# training loop
for epoch in range(total_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # move batch to GPU
        B = batch["input_ids"].shape[0] // 2                 # chosen/rejected split point

        policy_logits = policy_model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"]).logits
        with torch.no_grad():
            ref_logits = reference_model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"]).logits

        policy_logps = sequence_logps(policy_logits, batch["input_ids"], batch["loss_mask"])
        ref_logps = sequence_logps(ref_logits, batch["input_ids"], batch["loss_mask"])

        policy_chosen, policy_rejected = policy_logps[:B], policy_logps[B:]
        ref_chosen, ref_rejected = ref_logps[:B], ref_logps[B:]

        # -log sigmoid(beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected)))
        logits_diff = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        dpo_loss = -F.logsigmoid(beta * logits_diff).mean()
        train_loss = dpo_loss / gradient_accumulations          # normalized loss
        acc_loss += train_loss.item()                           # accumulated loss
        train_loss.backward()                                   # backward pass
                                                                 # accumulates gradients

        if step % gradient_accumulations == 0:
            # step
            grad_norm = clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # clip gradients
            optimizer.step()                                    # update weights
            optimizer.zero_grad()                               # zero out accumulated gradients
            scheduler.step()                                    # update LR
            global_step += 1                                    # count step

            # fraction of the batch where the policy ranks chosen over rejected
            # more than the reference did -- 0.5 is chance, 1.0 is perfect separation
            with torch.no_grad():
                reward_acc = (logits_diff > 0).float().mean().item()

            # log
            wandb.log({"train_loss": acc_loss,
                       "lr": scheduler.get_last_lr()[0],
                       "grad_norm": grad_norm.item(),
                       "reward_acc": reward_acc},
                       step=global_step)
            acc_loss = 0                                        # reset accumulation

        step += 1                                               # count forward pass

    # held-out DPO loss/accuracy on the preference test split
    policy_model.eval()                                         # eval mode
    with torch.no_grad():
        total_dpo_loss, total_correct, total_n = 0.0, 0, 0
        for b in test_dataloader:
            b = {k: v.to(device) for k, v in b.items()}         # move batch to GPU
            Bt = b["input_ids"].shape[0] // 2

            policy_logits = policy_model(input_ids=b["input_ids"],
                                         attention_mask=b["attention_mask"]).logits
            ref_logits = reference_model(input_ids=b["input_ids"],
                                         attention_mask=b["attention_mask"]).logits

            policy_logps = sequence_logps(policy_logits, b["input_ids"], b["loss_mask"])
            ref_logps = sequence_logps(ref_logits, b["input_ids"], b["loss_mask"])

            pc, pr = policy_logps[:Bt], policy_logps[Bt:]
            rc, rr = ref_logps[:Bt], ref_logps[Bt:]
            diff = (pc - pr) - (rc - rr)

            total_dpo_loss += -F.logsigmoid(beta * diff).sum().item()
            total_correct += (diff > 0).sum().item()
            total_n += Bt

    held_out_dpo_loss = total_dpo_loss / total_n
    held_out_dpo_acc = total_correct / total_n

    # held-out Dolly loss: did preference training degrade the Dolly fit?
    # (shared helper -- sft.py calls the same function on the SFT model)
    dolly_val_loss = evaluate_holdout_loss(policy_model, tokenizer, dolly_holdout, device,
                                           sft_config["model"]["max_length"],
                                           batch_size=config["training"]["batch_size"])

    wandb.log({"held_out_dpo_loss": held_out_dpo_loss,
               "held_out_dpo_acc": held_out_dpo_acc,
               "dolly_val_loss": dolly_val_loss,
               "epoch": epoch},
               step=global_step)

    policy_model.train()                                        # back to train mode

# push everything to the Hub, tagging the commit with what produced it
commit_message = (
    f"lr={config['training']['learning_rate']} beta={beta} "
    f"epochs={config['training']['num_epochs']} "
    f"wandb_run={wandb.run.url}"
)
policy_model.push_to_hub(config["outputs"]["hub_repo_id"],
                         private=config["outputs"]["hub_private"],
                         commit_message=commit_message)
tokenizer.push_to_hub(config["outputs"]["hub_repo_id"],
                      private=config["outputs"]["hub_private"],
                      commit_message=commit_message)
wandb.finish()
