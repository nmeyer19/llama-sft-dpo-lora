import yaml
from transformers import get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

from training.validation import split_dolly, tokenize_prompt_response, evaluate_holdout_loss

# load the config
with open("./configs/sft.yaml", "r") as file:
    config = yaml.safe_load(file)

# load + hold out the canonical Dolly split (dpo.py reuses this same split
# to check whether DPO training degrades the Dolly fit)
train_data, test_data = split_dolly(config)

# load base model and tokenizer
base_model, tokenizer = load_model(config)

train_tokenized = train_data.map(
    tokenize_prompt_response,
    fn_kwargs={"tokenizer": tokenizer, "max_length": config["model"]["max_length"]},
    remove_columns=train_data.column_names)

# construct the lora model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
)
lora_model = get_peft_model(base_model, lora_config)
# trainable params: 1,703,936 || all params: 1,237,518,336 || trainable%: 0.1377

# optimizer: AdamW
optimizer = torch.optim.AdamW(lora_model.parameters(), 
                              lr=config["training"]["learning_rate"])

# HF Seq2Seq collator for masked padding
collator = DataCollatorForSeq2Seq(tokenizer, lora_model, padding=True, 
                                  label_pad_token_id=-100)

# dataloader
train_dataloader = DataLoader(train_tokenized,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collator)

total_steps = ((len(train_dataloader) // 
               config["training"]["gradient_accumulation_steps"]) * 
               config["training"]["num_epochs"])

# lr scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config["training"]["warmup_steps"],
    num_training_steps=total_steps)

# train mode
lora_model.train()

# training-relevant initializations
acc_loss = 0
step = 1            # how many forward passess are performed
global_step = 0     # how many actual optimizer steps taken

total_epochs = config["training"]["num_epochs"]
gradient_accumulations = config["training"]["gradient_accumulation_steps"]

wandb.init(project=config["wandb"]["project"],
           name=config["wandb"]["run_name"],
           config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora_model = lora_model.to(device)

# just in case
optimizer.zero_grad()

# training loop
for epoch in range(total_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}         # move batch to GPU
        outputs = lora_model(**batch)                               # forward pass
        train_loss = outputs.loss / gradient_accumulations          # normalized loss
        acc_loss += train_loss.item()                               # accumulated loss
        train_loss.backward()                                       # backward pass
                                                                    # accumulates gradients

        if step % gradient_accumulations == 0:
            # step
            grad_norm = clip_grad_norm_(lora_model.parameters(), max_norm=1.0)  # clip gradients 
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # zero out accumulated gradients
            scheduler.step()                                        # update LR
            global_step += 1                                        # count step

            # log
            wandb.log({"train_loss": acc_loss, 
                       "lr": scheduler.get_last_lr()[0],
                       "grad_norm": grad_norm.item()},
                       step=global_step)
            acc_loss = 0                                            # reset accumulation
        
        step += 1                                                   # count forward pass

    # validate on held-out Dolly split (shared helper -- dpo.py calls the
    # same function on the policy model to check for Dolly-fit degradation)
    val_loss = evaluate_holdout_loss(lora_model, tokenizer, test_data, device,
                                     config["model"]["max_length"],
                                     batch_size=config["training"]["batch_size"])
    wandb.log({"val_loss": val_loss,
               "epoch": epoch},
               step=global_step)

    lora_model.train()                                              # back to train mode

# push everything to the Hub, tagging the commit with what produced it
commit_message = (
    f"lr={config['training']['learning_rate']} "
    f"epochs={config['training']['num_epochs']} "
    f"wandb_run={wandb.run.url}"
)
lora_model.push_to_hub(config["outputs"]["hub_repo_id"],
                       private=config["outputs"]["hub_private"],
                       commit_message=commit_message)
tokenizer.push_to_hub(config["outputs"]["hub_repo_id"],
                      private=config["outputs"]["hub_private"],
                      commit_message=commit_message)
wandb.finish()