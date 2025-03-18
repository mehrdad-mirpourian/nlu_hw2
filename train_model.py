"""
Code for Problem 1 of HW 2.
"""
import os 
import numpy as np
from collections.abc import Iterable

import torch
import torch.nn as nn

from typing import Dict, Any

from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, \
    BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, EvalPrediction

import evaluate
import optuna
from optuna.samplers import GridSampler
from optuna.pruners import MedianPruner

os.environ["WANDB_DISABLED"] = "true"

use_fp16 = torch.cuda.is_available()

import pickle

def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) -> Dataset:
    """
    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.
    """
    def tokenize_function(examples):
        """
        Tokenizes input text using Hugging Face BERT tokenizer.
        """
        if "text" not in examples or examples["text"] is None:
            examples["text"] = ["[EMPTY]"] * len(examples.get("label", [0] * len(examples)))

        examples["text"] = [t if isinstance(t, str) and t.strip() else "[EMPTY]" for t in examples["text"]]

        if not isinstance(examples["text"], list):
            examples["text"] = [str(examples["text"])]

        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    return dataset.map(tokenize_function, batched=True, batch_size=32)


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> BertForSequenceClassification:
    """
    Initializes a pre-trained model and applies BitFit if enabled.
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    for param in model.parameters():
        param.requires_grad = True  # Make all parameters trainable by default

    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False  # Freeze all non-bias parameters

    print("\nTrainable parameters after initializing model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    return model


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset, use_bitfit: bool = False) -> Trainer:
    """
    Creates and returns a Trainer for model training.
    """
    def compute_metrics(eval_pred: EvalPrediction):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    checkpoint_dir = "checkpoints_with_bitfit" if use_bitfit else "checkpoints_without_bitfit"   
                     
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        fp16=use_fp16
    )

    return Trainer(
        model_init=lambda: init_model(None, model_name, use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )


def hp_space(trial):
    """
    Defines the hyperparameter search space using Optuna.
    """
    lr = trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5])
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128])

    # Log the trial details
    print(f"Trial {trial.number}: lr={lr}, batch_size={batch_size}")

    return {
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
    }


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Returns keyword arguments passed to Trainer.hyperparameter_search.
    """
    def compute_objective(metrics):
        return metrics["eval_accuracy"]

    # Define the full grid search space
    search_space = {
        "learning_rate": [3e-4, 1e-4, 5e-5, 3e-5],
        "per_device_train_batch_size": [8, 16, 32, 64, 128],
    }

    # Use GridSampler to ensure all combinations are tested
    sampler = GridSampler(search_space)

    return {
        "direction": "maximize",
        "compute_objective": compute_objective,
        "n_trials": 2,  # Explicitly setting to 20
        "hp_space": hp_space,
        "sampler": sampler,  # Ensures every combination is tested
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

    # Save the best model with BitFit
    best_model_checkpoint = trainer_with_bitfit.state.best_model_checkpoint
    if best_model_checkpoint:
        best_model_with_bitfit = BertForSequenceClassification.from_pretrained(best_model_checkpoint)
        best_model_with_bitfit.save_pretrained("best_with_bitfit", safe_serialization=True)
    else:
        print("Warning: No best model checkpoint found for BitFit!")

    # Save results
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best_with_bitfit, f)

    # Fine-tune **without BitFit**
    trainer_without_bitfit = init_trainer(model_name, imdb["train"], imdb["val"], use_bitfit=False)
    best_without_bitfit = trainer_without_bitfit.hyperparameter_search(**hyperparameter_search_settings())

    # Print best checkpoint path to verify if it was saved correctly
    print(f"Best checkpoint for with BitFit: {trainer_with_bitfit.state.best_model_checkpoint}")
    print(f"Best checkpoint for without BitFit: {trainer_without_bitfit.state.best_model_checkpoint}")

    # Save the best model without BitFit
    best_model_checkpoint = trainer_without_bitfit.state.best_model_checkpoint
    if best_model_checkpoint:
        best_model_without_bitfit = BertForSequenceClassification.from_pretrained(best_model_checkpoint)
        best_model_without_bitfit.save_pretrained("best_without_bitfit", safe_serialization=True)
    else:
        print("Warning: No best model checkpoint found for Non-BitFit!")

    # Save results
    with open("train_results_without_bitfit.p", "wb") as f:
        pickle.dump(best_without_bitfit, f)
