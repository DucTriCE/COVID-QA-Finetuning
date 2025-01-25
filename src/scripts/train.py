import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
from data.dataset import COVIDQADataset
from models.model import COVIDQAModel
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator
from utils.utils import compute_metrics, load_config, preprocess_training_examples, preprocess_validation_examples
from functools import partial


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = load_config("config/cfg.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    raw_dataset = load_dataset(config["dataset_name"])

    raw_train_dataset = raw_dataset["train"].map(
        lambda example: {
            "context": " ".join(example["context_chunks"]),  # Join context chunks into a single string
            "answers": {
                "text": [example["answer"]],  # Format the answer
                "answer_start": [" ".join(example["context_chunks"]).find(example["answer"])],  # Find the answer start position
            },
        },
        remove_columns=["context_chunks", "answer"],  # Remove unnecessary columns
    )
    raw_train_dataset.filter(lambda x: len(x["answers"]["text"]) != 1)

    raw_val_dataset = raw_dataset["validation"].map(
        lambda example: {
            "context": " ".join(example["context_chunks"]),  # Join context chunks into a single string
            "answers": {
                "text": [example["answer"]],  # Format the answer
                "answer_start": [" ".join(example["context_chunks"]).find(example["answer"])],  # Find the answer start position
            },
        },
        remove_columns=["context_chunks", "answer"],  # Remove unnecessary columns
    )

    train_dataset = raw_train_dataset.map(
        lambda examples: preprocess_training_examples(examples, tokenizer, config),
        batched=True,
        remove_columns=raw_train_dataset.column_names,
    )
    val_dataset = raw_val_dataset.map(
        lambda examples: preprocess_validation_examples(examples, tokenizer, config),
        batched=True,
        remove_columns=raw_val_dataset.column_names,
    )

    model = COVIDQAModel()

    # Start training
    model.train_net(train_dataset, val_dataset)
    model.validation(val_dataset, raw_val_dataset)
