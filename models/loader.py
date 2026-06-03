import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel

def load_model(
    config: dict, 
    checkpoint_path: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    
    cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    tokenizer.pad_token = tokenizer.eos_token   # no padding by default w llama

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], 
        torch_dtype=getattr(torch, cfg["torch_dtype"]),
        device_map="auto"
    )

    if checkpoint_path:
        lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_lora = lora_model.merge_and_unload()
        return (merged_lora, tokenizer)
    else:
        return (base_model, tokenizer)