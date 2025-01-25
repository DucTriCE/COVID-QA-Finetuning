from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import logging
from utils.utils import load_config, compute_metrics, c_m
import evaluate     

class COVIDQAModel:
    def __init__(self, config_path="config/cfg.yaml"):
        self.config = load_config(config_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.metric = evaluate.load(self.config["metric"])
    
        self.training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            evaluation_strategy=self.config["evaluation_strategy"],
            learning_rate=float(self.config["learning_rate"]),
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            num_train_epochs=self.config["num_train_epochs"],
            dataloader_num_workers=self.config["dataloader_num_workers"],
            weight_decay=self.config["weight_decay"],
            save_strategy=self.config["save_strategy"],
            warmup_steps=self.config["warmup_steps"],
            fp16=self.config["fp16"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
            logging_dir=self.config["logging_dir"],
            logging_strategy=self.config["logging_strategy"],
            eval_on_start=self.config["eval_on_start"],
        )

        self.evaluating_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            dataloader_num_workers=self.config["dataloader_num_workers"],
            logging_dir=self.config["logging_dir"],
        )
            

    def train_net(self, train_dataset, val_dataset):
        """
        Train the model using the provided DataLoader.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader for training data.
            eval_dataloader (DataLoader): PyTorch DataLoader for evaluation data.
        """


        # Define the Trainer
        trainer = Trainer(
            model=self.model,
            args= self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        # Train the model
        trainer.train()

        # Save the model and tokenizer
        self.model.save_pretrained(self.config["output_dir"])
        self.tokenizer.save_pretrained(self.config["output_dir"])

    def validation(self, val_dataset, raw_val_dataset):
        """
        Validate the model using the provided DataLoader.

        Args:
            eval_dataloader (DataLoader): PyTorch DataLoader for evaluation data.
        """

        trainer = Trainer(
            model=self.model,
            args= self.evaluating_args,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        predictions, _, _ = trainer.predict(val_dataset)
        start_logits, end_logits = predictions
        print(compute_metrics(self.metric, start_logits, end_logits, val_dataset, raw_val_dataset, self.config))