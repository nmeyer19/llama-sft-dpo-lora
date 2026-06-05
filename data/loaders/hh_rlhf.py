from datasets import load_dataset
from data.loaders.base import BaseDataLoader

class HHRLHFDataLoader(BaseDataLoader):
    """
    Data loader for the HH-RLHF dataset.
    Loads the harmless-base subset of the HH-RLHF dataset and formats it into
    a triple of (prompt, chosen, rejected) for DPO training.
    """

    def load(self) -> None:
        """Load and preprocess the HH-RLHF dataset."""
        cfg = self.config["data"]
        raw_dataset = load_dataset(
            cfg["dataset"],
            data_dir=cfg["subset"],
            split=cfg["split"]
        )

        if cfg["max_samples"] is not None:
            raw_dataset = raw_dataset.shuffle(seed=cfg["seed"]).select(range(cfg["max_samples"]))
        
        self.data = raw_dataset.map(self._format_triple)
    
    def _format_triple(self, example: dict) -> dict:
        """
        Formats a raw HH-RLHF example into a (prompt, chosen, rejected) triple.

        HH-RLHF stores conversations with Human/Assistant turns concatenated.
        We split on the last Assistant turn to separate the prompt from the responses.

        Raises:
            ValueError: If the expected split token is not found in the example.
        """
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        split_token = "\n\nAssistant:"
        if split_token not in chosen or split_token not in rejected:
            raise ValueError(f"Expected split token '{split_token}' not found in example: {chosen}")

        prompt = chosen.rsplit(split_token, 1)[0] + split_token
        chosen_resp = chosen.rsplit(split_token, 1)[1].strip()
        rejected_resp = rejected.rsplit(split_token, 1)[1].strip()

        return {
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        }