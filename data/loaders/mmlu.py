from datasets import load_dataset, concatenate_datasets
from data.loaders.base import BaseDataLoader

class MMLUDataLoader(BaseDataLoader):
    """
    Data loader for the MMLU benchmark.
    Loads 50 examples from each subject and formats them into a multiple-choice
    style question for evaluation.
    """

    def load(self) -> None:
        """Load and preprocess the MMLU dataset."""
        cfg = self.config["benchmark"]
        raw_dataset = load_dataset(cfg["dataset"], "all", split=cfg["split"])

        if cfg["max_samples"] is not None:
            samps_per_subj = cfg["max_samples"] // len(cfg["subjects"])
            subject_datasets = []

            for subject in cfg["subjects"]:
                subject_data = raw_dataset.filter(lambda x, s=subject: x["subject"] == s)
                subject_data = subject_data.shuffle(seed=cfg["seed"])
                subject_data = subject_data.select(range(samps_per_subj))
                subject_datasets.append(subject_data)
            
            full_dataset = concatenate_datasets(subject_datasets)

        else:
            full_dataset = raw_dataset

        self.data = full_dataset.map(self._format_mcq)

    def _format_mcq(self, example: dict) -> dict:
        """
        Formats a raw MMLU example into a multiple-choice question.
        Each example contains a question, subject, choices, and answer field.
        We Concatenate the question and choices fields and return the desired 
        MCQ format.
        """

        question = example["question"]
        choices = example["choices"]
        
        prompt = (
            f"Question: {question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            f"Answer:"
        )
        
        answer = ["A", "B", "C", "D"][example["answer"]]

        return {
            "prompt": prompt,
            "answer": answer,
        }