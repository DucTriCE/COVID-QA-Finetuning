from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from utils.utils import load_config, preprocess_training_examples, preprocess_validation_examples

class COVIDQADataset(Dataset):
    """
    COVID-QA dataset.

    Args:
        split (str): Dataset split to load ("train" or "validation").
        config_path (str): Path to the configuration file (default: "config/config.yaml").
    """

    def __init__(self, split, config_path="config/cfg.yaml"):
        self.config = load_config(config_path)
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.dataset = self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Load and preprocess the dataset.

        Returns:
            dataset (torch.utils.data.Dataset): Preprocessed dataset.
        """
        dataset = load_dataset(self.config["dataset_name"], split=self.split)
        
        if self.split == "train":
            tokenized_dataset = dataset.map(
                lambda examples: preprocess_training_examples(examples, self.tokenizer, self.config), 
                batched=True, 
                remove_columns = dataset.column_names
            )
        else:   
            tokenized_dataset = dataset.map(
                lambda examples: preprocess_validation_examples(examples, self.tokenizer, self.config),
                batched=True, 
                remove_columns = dataset.column_names)

        return tokenized_dataset

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input IDs.
                - attention_mask (torch.Tensor): Attention mask.
                - start_positions (int): Start position of the answer.
                - end_positions (int): End position of the answer.
        """
        sample = self.dataset[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(sample["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(sample["end_positions"], dtype=torch.long),
        }