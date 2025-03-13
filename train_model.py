"""
Code for Problem 1 of HW 2.
"""
import os
import numpy as np
from collections.abc import Iterable
from typing import Dict, Any

import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, \
    BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, EvalPrediction

import evaluate
import optuna
from optuna.samplers import GridSampler

os.environ["WANDB_DISABLED"] = "true"

use_fp16 = torch.cuda.is_available()

import pickle



def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    def tokenize_function(examples):
        """
        Tokenizes input text using Hugging Face BERT tokenizer.
        Handles multiple edge cases to ensure robust preprocessing.
        """
        # Edge Case: Handle missing or empty text
        if "text" not in examples or examples["text"] is None:
            examples["text"] = ["[EMPTY]"] * len(examples.get("label", [0] * len(examples)))

        # Edge Case: Handle empty strings or unexpected formats
        examples["text"] = [t if isinstance(t, str) and t.strip() else "[EMPTY]" for t in examples["text"]]

        # Edge Case: Handle different data structures (alternative keys)
        if not isinstance(examples["text"], list):
            examples["text"] = [str(examples["text"])]

        try:
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",  # Ensures all sequences are of equal length
                truncation=True,       # Ensures long sequences are truncated
                max_length=512        # As per Appendix A.2 of BERT paper
            )
        except Exception as e:
            # Edge Case: Handle tokenization failures
            print(f"Tokenization error: {e}")
            return {
                "input_ids": [[0] * 512],  # Default empty tokens
                "token_type_ids": [[0] * 512],
                "attention_mask": [[0] * 512]
            }

        return tokenized

    # Edge Case: Handle large batch sizes by limiting batch processing
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=32)

    return tokenized_dataset


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    # Ensure default behavior when use_bitfit=False: I added this line
    # Becasue during testing my code I got some unexpected output when
    # bitfit = False. Adding this line made my output behave correctly.
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    for param in model.parameters():
        param.requires_grad = True  # Make all parameters trainable by default

    # If BitFit is enabled, freeze all non-bias parameters
    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:  # Freeze all non-bias parameters
                param.requires_grad = False

    return model


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    # Step 1: Define a function for computing accuracy on validation
    def compute_metrics(eval_pred: EvalPrediction):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Step 2: Define training arguments for our trainer
    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=use_fp16  # Enable mixed-precision only if GPU is available
    )

    # Step 3: Create and return trainer object with model_init for hyperparameter tuning
    trainer = Trainer(
        model_init=lambda: init_model(None, model_name, use_bitfit),  # Allow hyperparameter search
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )
    return trainer

def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    # Define the optimized hyperparameter search space
    search_space = {
        "learning_rate": [2e-5, 3e-5, 5e-5],  # Typical learning rates for transformers
        "per_device_train_batch_size": [8, 16],  # Corrected batch size options
        "weight_decay": [0.01, 0.001],  # Regularization values
        "num_train_epochs": [2, 3],  # Reduced max epochs to speed up tuning
        "dropout": [0.1, 0.2],  # Only 2 dropout choices
        "optimizer": ["adamw_torch"],  # Only AdamW, since it's standard for Transformers
        "seed": [42],       
    }

    # Use GridSampler for predefined search space
    grid_sampler = GridSampler(search_space, seed=42)

    # Function to maximize accuracy
    def compute_objective(metrics):
        return metrics["eval_accuracy"]

    return {
        "direction": "maximize",
        "compute_objective": compute_objective,
        "n_trials": len(search_space["learning_rate"]) *
                    len(search_space["per_device_train_batch_size"]) *
                    len(search_space["weight_decay"]) *
                    len(search_space["num_train_epochs"]) *
                    len(search_space["dropout"]) *
                    len(search_space["optimizer"]),
        "sampler": grid_sampler,
    }

if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Fine-tune **with BitFit**
    trainer_with_bitfit = init_trainer(model_name, imdb["train"], imdb["val"], use_bitfit=True)
    best_with_bitfit = trainer_with_bitfit.hyperparameter_search(**hyperparameter_search_settings())

    # Save results
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best_with_bitfit, f)

    # Fine-tune **without BitFit**
    trainer_without_bitfit = init_trainer(model_name, imdb["train"], imdb["val"], use_bitfit=False)
    best_without_bitfit = trainer_without_bitfit.hyperparameter_search(**hyperparameter_search_settings())

    # Save results
    with open("train_results_without_bitfit.p", "wb") as f:
        pickle.dump(best_without_bitfit, f)
