import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import COVIDQAModel
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from utils.utils import compute_metrics, load_config, preprocess_validation_examples
import evaluate
import os
import torch

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = load_config("config/cfg.yaml")
    tokenizer = AutoTokenizer.from_pretrained('fine-tuned-model/checkpoint-19500')
    model = AutoModelForQuestionAnswering.from_pretrained('fine-tuned-model/checkpoint-19500')

    metric = evaluate.load(config["metric"])

    raw_dataset = load_dataset(config["dataset_name"])
    

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

    raw_val_dataset = raw_val_dataset.filter(lambda x: len(x["answers"]["text"]) == 1)


    val_dataset = raw_val_dataset.map(
        lambda examples: preprocess_validation_examples(examples, tokenizer, config),
        batched=True,
        remove_columns=raw_val_dataset.column_names,
    )

    evaluating_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        dataloader_num_workers=config["dataloader_num_workers"],
        logging_dir=config["logging_dir"],
    )

    trainer = Trainer(
        model=model,
        args= evaluating_args,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    predictions, _, _ = trainer.predict(val_dataset)
    start_logits, end_logits = predictions
    print(compute_metrics(metric, start_logits, end_logits, val_dataset, raw_val_dataset, config))

