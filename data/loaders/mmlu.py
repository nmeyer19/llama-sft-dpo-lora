from datasets import load_dataset, concatenate_datasets
from data.loaders.base import BaseDataLoader
from typing import Any

class MMLUDataLoader(BaseDataLoader):
    """
    Data loader for the MMLU benchmark.
    Loads 50 examples from each subject and configures few-shot prompting.
    """

    def load(self) -> None:
        """
        Load and preprocess the MMLU dataset.
        Add few-shot prompting using the dev pool examples from the dataset.
        """
        # load and preprocess
        cfg = self.config["benchmark"]
        raw_dataset = load_dataset(cfg["dataset"], "all", split=cfg["split"])

        if cfg["max_samples"] is not None:
            samps_per_subj = cfg["max_samples"] // len(cfg["subjects"])
            subject_datasets = []

            for subject in cfg["subjects"]:
                subject_data = raw_dataset.filter(lambda x, s=subject: x["subject"] == s)
                subject_data = subject_data.shuffle(seed=cfg["seed"])
                subject_data = subject_data.select(range(min(samps_per_subj, len(subject_data))))
                subject_datasets.append(subject_data)
            
            full_dataset = concatenate_datasets(subject_datasets)

        else:
            full_dataset = raw_dataset

        self.data = full_dataset.map(self._format_mcq)

        # fewshot formatting
        dev_raw = load_dataset(cfg["dataset"], "all", split="dev")
        num_shots = cfg["num_shots"]
        self.fewshot = {}

        for subject in cfg["subjects"]:
            subj_dev = dev_raw.filter(lambda x, s=subject: x["subject"] == s)
            subj_dev = subj_dev.select(range(min(num_shots, len(subj_dev))))
            self.fewshot[subject] = list(subj_dev.map(self._format_mcq))


    def _format_mcq(self, example: dict) -> dict:
        """
        Converts integer answer to a letter and passes question, choices, and
        subject unchanged.
        """

        return {
            "subject": example["subject"],
            "question": example["question"],
            "choices": example["choices"],
            "answer": ["A", "B", "C", "D"][example["answer"]]
        }

    def get_fewshot(self) -> Any:
        """Return the processed data. Raises if load() has not been called."""
        if not hasattr(self, "fewshot"):
            raise RuntimeError(
                f"Fewshot not loaded. Call {self.__class__.__name__}.load() first.")
        return self.fewshot