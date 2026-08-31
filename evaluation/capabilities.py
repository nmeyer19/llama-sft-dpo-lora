# load configs
import yaml
from models.loader import load_model
from evaluation.benchmarks.mmlu import MMLUBenchmark
import torch

with open("./configs/sft.yaml", "r") as file:
    sft_config = yaml.safe_load(file)

with open("./configs/dpo.yaml", "r") as file:
    dpo_config = yaml.safe_load(file)

with open("./configs/mmlu.yaml", "r") as file:
    mmlu_config = yaml.safe_load(file)

# define the models being evaluated
models = [
    {"name": "base-model", 
     "config": sft_config, 
     "checkpoint": None},
    {"name": "sft-model", 
     "config": sft_config, 
     "checkpoint": sft_config["outputs"]["model_dir"]},
    #{"name": "dpo-model", 
    #"config": dpo_config, 
    #"checkpoint": dpo_config["outputs"]["model_dir"]},
]

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set base results_dir for proper directory saving
gen_results_dir = mmlu_config["outputs"]["results_dir"]

for m in models:
    # kick things off and reconfigure output location
    print(f"\nevaluating {m['name']}...")
    mmlu_config["outputs"]["results_dir"] = gen_results_dir + "/" + m["name"]
    mmlu_config["outputs"]["prompt_style"] = "plain" if m["name"] == "base-model" else "dolly"

    # load model and put it on the gpu
    model, tokenizer = load_model(m["config"], checkpoint_path=m["checkpoint"])
    model = model.to(device)

    # create MMLUBenchmark object and run eval
    benchmark = MMLUBenchmark(mmlu_config, model, tokenizer)
    benchmark.run()

    # print results
    print(benchmark.results)

    # free up GPU for next turn
    del model, tokenizer, benchmark
    torch.cuda.empty_cache()
