import yaml
from data.loaders.dolly import DollyDataLoader
from transformers import get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

# load the config
with open("./configs/sft.yaml", "r") as file:
    config = yaml.safe_load(file)

# load training data
dataloader = DollyDataLoader(config)
dataloader.load()
dataset = dataloader.get_data()

# hold some out to test
split_dataset = dataset.train_test_split(test_size=0.05, seed=config["data"]["seed"])
train_data = split_dataset["train"]
test_data = split_dataset["test"]

# load base model and tokenizer
base_model, tokenizer = load_model(config)

# tokenizes an example and masks the prompt tokens
def tokenize(example):
    # tokenize prompt and response separately
    tokenized_prompt = tokenizer(example["prompt"], padding=False)
    
    tokenized_response = tokenizer(example["response"], padding=False, 
                                   add_special_tokens=False)

    # concatenate token ids and truncate to max_length config
    prompt_ids = tokenized_prompt["input_ids"] 
    response_ids = tokenized_response["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    input_ids = input_ids[:config["model"]["max_length"]]

    # mask prompt tokens for no gradient signal and truncate length
    # and truncate to max_length config
    labels = [-100]*len(prompt_ids) + response_ids
    labels = labels[:config["model"]["max_length"]]

    # attention_mask all 1s (inherits truncation from labels)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_tokenized = train_data.map(tokenize, remove_columns=train_data.column_names)
test_tokenized = test_data.map(tokenize, remove_columns=test_data.column_names)

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

# dataloaders
train_dataloader = DataLoader(train_tokenized, 
                              batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collator)

test_dataloader = DataLoader(test_tokenized, 
                              batch_size=config["training"]["batch_size"],
                              shuffle=False, collate_fn=collator)

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

    # validate on held-out
    lora_model.eval()                                               # eval mode
    with torch.no_grad():
        total_loss, total_tokens = 0.0, 0
        for b in test_dataloader:
            b = {k: v.to(device) for k, v in b.items()}             # move batch to GPU
            outputs = lora_model(**b)                               # forward pass
            n_tokens = (b["labels"] != -100).sum()                  # ignore prompt tokens
            total_loss += outputs.loss.float() * n_tokens
            total_tokens += n_tokens 

    # get total loss and log
    val_loss = (total_loss / total_tokens).item()                   
    wandb.log({"val_loss": val_loss, 
               "epoch": epoch},
               step=global_step)

    lora_model.train()                                              # back to train mode

# save everything
lora_model.save_pretrained(config["outputs"]["model_dir"])
tokenizer.save_pretrained(config["outputs"]["model_dir"])
wandb.finish()