import yaml
from data.loaders.dolly import DollyDataLoader
from transformers import AutoTokenizer
import models.loader

# load the config
with open("./configs/sft.yaml", "r") as file:
    config = yaml.safe_load(file)

# load training data
dataloader = DollyDataLoader(config)
dataloader.load()
dataset = dataloader.get_data()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

# tokenizes an example and masks the prompt tokens
def tokenize(example):
    # tokenize full example text
    example_text = example["prompt"] + example["response"]
    tokenized_ex = tokenizer(example_text, truncation=True, 
                             max_length=config["model"]["max_length"],
                             padding=False)
    # mask prompt tokens for no gradient signal
    prompt_len = len(tokenizer(example["prompt"], truncation=True,
                     max_length=config["model"]["max_length"],
                     padding=False)["input_ids"])
    labels = tokenized_ex["input_ids"].copy()
    for i in range(prompt_len):
        labels[i] = -100
    # create a new field indicating which tokens to ignore in gradient calcs
    tokenized_ex["labels"] = labels 
    
    return tokenized_ex

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)



"""
3a. load the base model with models/loader.py and load the lora adapters w PEFT
- get_peft_model() to load
- model.print_trainable_parameters() to check 
3b. wrap the base model with the lora adapters for SFT       
4a. set up the optimizer, LR schedulder + warmup
- remember we're doing 4 gradient accumulation steps
4b. write the training loop itself (foward pass, loss, backward pass, 
optimizer.step)
- add padding to concatenated sequence: DataLoader with a collator for padding 
and mask padding tokens
- for each batch, loss = forward_pass(batch) / gradient_accumulation_steps
before calling loss.backward()
- only if step % gradient_accumulation_steps == 0 then optimizer.step() and
optimizer.zero_grad()
4c. within the training loop, log to wandb                  
5. save final lora adapter matrices/weights 
- model.save_pretrained(output_dir) - saves only the adapter matrices and the
lora config
"""