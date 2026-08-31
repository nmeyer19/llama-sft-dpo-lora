from data.loaders.mmlu import MMLUDataLoader
from evaluation.benchmarks.base import BaseBenchmark
import torch

class MMLUBenchmark(BaseBenchmark):
    """
    MMLU benchmark object.
    Implements load() and evaluate() functions for MMLU.
    Inherits run() and save_results() implementations from BaseBenchmark ABC.
    Runs the full MMLU eval pipeline through saving results via the run() 
    function call.
    """

    def load(self) -> None: 
        dataloader = MMLUDataLoader(self.config)
        dataloader.load()
        self.data = dataloader.get_data()

    def evaluate(self) -> None:
        """Evaluate the model on the MMLU benchmark."""
        
        # get IDs for target tokens
        id_a = self.tokenizer.encode(" A", add_special_tokens=False)[-1]
        id_b = self.tokenizer.encode(" B", add_special_tokens=False)[-1]
        id_c = self.tokenizer.encode(" C", add_special_tokens=False)[-1]
        id_d = self.tokenizer.encode(" D", add_special_tokens=False)[-1]

        # infer device from model
        device = next(self.model.parameters()).device

        # response tracker - list of dicts
        self.responses = []
        
        # subject counters for self.results
        subj_corr = {}
        subj_total = {}

        # evaluation loop
        self.model.eval()
        batch_size = self.config["evaluation"]["batch_size"]

        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                # create batch
                batch = self.data.select(range(i, min(i + batch_size, 
                                                      len(self.data))))
                # tokenize prompts and move to device
                tokenized_prompts = self.tokenizer(list(batch["prompt"]), 
                                                   return_tensors="pt",
                                                   padding=True, 
                                                   truncation=True)
                tokenized_prompts = {k: v.to(device) for k, v in 
                                     tokenized_prompts.items()}

                # forward pass
                outputs = self.model(**tokenized_prompts)
                # outputs.logits.shape: [batch_size, seq_length, vocab_size]
                
                # sum prompts along seq_length dimension to get a tensor of 
                # sequence lengths per example and subtract 1 to get the index
                # of the last valid token
                seq_lengths = torch.sum(tokenized_prompts["attention_mask"], 
                                        dim=1)
                last_token_indices = seq_lengths - 1

                # tensor of indices to access each example in the batch
                batch_indices = torch.arange(len(batch), device=device)

                # index into outputs.logits via the batch_indices in dim=0
                # and the last_token_indices in dim=1
                # to get a tensor of [batch_size, vocab_size]
                # i.e. a tensor of the logits over the model's vocabulary
                # for the next token following the prompt for each example
                # in the batch
                last_token_logits = outputs.logits[batch_indices, 
                                                   last_token_indices, :]

                # get the logits for just the 4 tokens we care about
                # [batch_size, 4] and argmax -> [batch_size] (index of predictions)
                mc_logits = last_token_logits[:, [id_a, id_b, id_c, id_d]]
                pred_ids = torch.argmax(mc_logits, dim=1)

                # map pred_id back to letter
                pred_ids = pred_ids.tolist()
                predictions = [["A", "B", "C", "D"][idx] for idx in pred_ids]

                subjects = list(batch["subject"])   # list of batch subjects
                prompts = list(batch["prompt"])     # list of batch prompts
                answers = list(batch["answer"])     # list of batch answers
                
                for sj, pt, an, pd in zip(subjects, prompts, answers, predictions):
                    is_correct = an == pd
                    self.responses.append({"subject": sj,
                                           "prompt": pt,
                                           "answer": an,
                                           "prediction": pd,
                                           "correct": is_correct})
                    
                    if is_correct: subj_corr[sj] = subj_corr.get(sj, 0) + 1
                    subj_total[sj] = subj_total.get(sj, 0) + 1

        subj_acc = {s: subj_corr.get(s, 0) / subj_total[s] for s in subj_total}
        total_acc = sum(subj_corr.values()) / sum(subj_total.values())

        self.results = {"per_subject": subj_acc, "total": total_acc}