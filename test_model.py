"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    # Load the fine-tuned model from the directory
    model = BertForSequenceClassification.from_pretrained(directory)

    # Define evaluation arguments (no training, only testing)
    training_args = TrainingArguments(
        output_dir="./results",  # Store evaluation results
        per_device_eval_batch_size=16,  # Match batch size used in training
        do_train=False,  # Ensure training is disabled
        do_eval=True,  # Enable evaluation
        logging_dir="./logs",  # Directory for logs
        report_to="none"  # Disable unnecessary logging (e.g., wandb)
    )

    # Define compute_metrics function (same as in training)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Create Trainer for testing
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )

    return trainer


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    # tester = init_tester("path_to_your_best_model")
    tester_with_bitfit = init_tester("best_with_bitfit")
    tester_without_bitfit = init_tester("best_without_bitfit")

    # Test
    # results = tester.predict(imdb["test"])
    # with open("test_results.p", "wb") as f:
    #     pickle.dump(results, f)

    results_with_bitfit = tester_with_bitfit.predict(imdb["test"])
    with open("test_results_with_bitfit.p", "wb") as f:
        pickle.dump(results_with_bitfit, f)

results_without_bitfit = tester_without_bitfit.predict(imdb["test"])
with open("test_results_without_bitfit.p", "wb") as f:
    pickle.dump(results_without_bitfit, f)
