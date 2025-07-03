import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def load_jsonl(path):
    """Load a .jsonl file into a pandas DataFrame."""
    return pd.read_json(path, lines=True)

def evaluate_cold_start(pred_file, train_df, user_df):
    """
    Group users in the test set by their training interaction frequency,
    then compute MSE, MAE, and Count for each frequency group.

    Parameters:
    - pred_file: path to JSONL file with prediction records
    - train_df: pandas DataFrame of the training set
    - user_df: pandas DataFrame of user metadata

    Returns:
    - summary_lines: a list of formatted result strings
    """

    # Compute training frequency for each user
    user_train_freq = train_df.groupby("userID").size().reindex(user_df["userID"], fill_value=0)
    user_df["training_frequency"] = user_train_freq

    # Load prediction file: each line should be a JSON with userID, true_rating, pred_rating
    with open(pred_file, "r") as f:
        predictions = [json.loads(line) for line in f]

    # Initialize results by training frequency
    group_results = defaultdict(lambda: {"squared_error": [], "abs_error": []})

    for pred in tqdm(predictions, desc="Evaluating"):
        uid = pred["userID"]
        true_rating = pred["true_rating"]
        pred_rating = pred["pred_rating"]

        # Skip unknown users
        freq = user_df[user_df["userID"] == uid]["training_frequency"].values
        if len(freq) == 0:
            continue
        freq = int(freq[0])

        # Record errors
        group_results[freq]["squared_error"].append((pred_rating - true_rating) ** 2)
        group_results[freq]["abs_error"].append(abs(pred_rating - true_rating))

    # Print summary results
    summary = []
    for freq in sorted(group_results.keys()):
        mse = np.mean(group_results[freq]["squared_error"]) if group_results[freq]["squared_error"] else float("nan")
        mae = np.mean(group_results[freq]["abs_error"]) if group_results[freq]["abs_error"] else float("nan")
        count = len(group_results[freq]["squared_error"])
        line = f"Training Frequency {freq}: MSE = {mse:.4f}, MAE = {mae:.4f}, Count = {count}"
        print(line)
        summary.append(line)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the prediction file (.jsonl)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data (.jsonl)")
    parser.add_argument("--user_file", type=str, required=True, help="Path to the user metadata (.jsonl)")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the output summary")

    args = parser.parse_args()

    train_df = load_jsonl(args.train_file)
    user_df = load_jsonl(args.user_file)
    summary_lines = evaluate_cold_start(args.pred_file, train_df, user_df)

    # Optionally save results to file
    if args.output_file:
        with open(args.output_file, "w") as f:
            for line in summary_lines:
                f.write(line + "\n")

