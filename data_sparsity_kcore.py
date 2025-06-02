# data_sparsity_kcore.py

import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import html
import json

# Fixed parameters
MAX_USER_REVIEWS = 15
MAX_ITEM_REVIEWS = 15
RANDOM_SEED_TRAIN = 984444699
RANDOM_SEED_TEST = 636893143
TRAIN_RATE = 0.8

def clean_line(line):
    return ''.join(c for c in line if c.isprintable())

def filter_k_core(df, k):
    """
    Iteratively filter users/items with less than k interactions.
    """
    while True:
        item_counts = df['itemID'].value_counts()
        user_counts = df['userID'].value_counts()
        df = df[df['itemID'].isin(item_counts[item_counts >= k].index)]
        df = df[df['userID'].isin(user_counts[user_counts >= k].index)]

        if len(item_counts) == len(df['itemID'].value_counts()) and \
           len(user_counts) == len(df['userID'].value_counts()):
            break
    return df

def unescape_reviews(df):
    df["reviewText"] = df["reviewText"].apply(html.unescape)
    return df

def preprocess(df, n_users, n_items):
    """
    Assign reviewIDs and prepare user/item review lists for model use.
    """
    df = df.assign(reviewID=list(range(df.shape[0])))
    user_df = pd.DataFrame({"userID": range(n_users), "reviewIDs": [""] * n_users})
    item_df = pd.DataFrame({"itemID": range(n_items), "reviewIDs": [""] * n_items})

    user_reviews = [[] for _ in range(n_users)]
    item_reviews = [[] for _ in range(n_items)]

    for i, row in df.iterrows():
        user_reviews[row["userID"]].append(i)
        item_reviews[row["itemID"]].append(i)

    for i in range(n_users):
        selected = random.sample(user_reviews[i], min(MAX_USER_REVIEWS, len(user_reviews[i])))
        user_df.at[i, "reviewIDs"] = ",".join(map(str, selected))

    for i in range(n_items):
        selected = random.sample(item_reviews[i], min(MAX_ITEM_REVIEWS, len(item_reviews[i])))
        item_df.at[i, "reviewIDs"] = ",".join(map(str, selected))

    return df, user_df, item_df

def split_and_save_data(df, output_dir):
    n_users = df["userID"].nunique()
    n_items = df["itemID"].nunique()

    # Train/valid/test split
    train, valid = train_test_split(df, train_size=TRAIN_RATE, random_state=RANDOM_SEED_TRAIN)
    valid, test = train_test_split(valid, test_size=0.5, random_state=RANDOM_SEED_TEST)

    def remove_overlap(base, other):
        return base[~base[["userID", "itemID"]].apply(tuple, axis=1).isin(set(zip(other["userID"], other["itemID"])))]

    train = remove_overlap(train, valid)
    train = remove_overlap(train, test)
    valid = remove_overlap(valid, test)

    train, user_df, item_df = preprocess(train, n_users, n_items)
    train = unescape_reviews(train)
    valid = unescape_reviews(valid)
    test = unescape_reviews(test)

    # Remove empty user/item in test/valid
    empty_users = user_df[user_df["reviewIDs"] == ""]["userID"].tolist()
    empty_items = item_df[item_df["reviewIDs"] == ""]["itemID"].tolist()
    valid = valid[~valid["userID"].isin(empty_users) & ~valid["itemID"].isin(empty_items)]
    test = test[~test["userID"].isin(empty_users) & ~test["itemID"].isin(empty_items)]

    os.makedirs(output_dir, exist_ok=True)
    train.to_json(os.path.join(output_dir, "train_raw.jsonl"), orient="records", lines=True)
    valid.to_json(os.path.join(output_dir, "valid_raw.jsonl"), orient="records", lines=True)
    test.to_json(os.path.join(output_dir, "test_raw.jsonl"), orient="records", lines=True)
    user_df.to_json(os.path.join(output_dir, "user.jsonl"), orient="records", lines=True)
    item_df.to_json(os.path.join(output_dir, "item.jsonl"), orient="records", lines=True)

def process_kcore(json_file, output_root, k_values):
    """
    Process a raw .json review file and apply multiple k-core filters.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = [clean_line(line) for line in f if line.strip()]
        records = [json.loads(line) for line in lines if line.strip()]

    df = pd.DataFrame(records)
    df = df[df["reviewText"] != ""]
    df.drop(columns=["reviewerName", "reviewTime", "unixReviewTime", "helpful", "summary"], errors="ignore", inplace=True)
    df["reviewerID"] = df["reviewerID"].astype("category").cat.codes
    df["asin"] = df["asin"].astype("category").cat.codes
    df.rename(columns={"reviewerID": "userID", "asin": "itemID", "overall": "rating"}, inplace=True)

    for k in k_values:
        out_dir = os.path.join(output_root, f"{k}-core")
        k_df = filter_k_core(df.copy(), k)
        if k_df.empty:
            print(f"Skipping {k}-core (no data)")
            continue
        print(f"Generating {k}-core: {k_df.shape[0]} samples")
        split_and_save_data(k_df, out_dir)

if __name__ == "__main__":
    input_path = "data/raw_reviews"  # Replace with your input dir
    output_path = "data/sparsity_kcore"
    k_values = [0, 3, 5, 8, 10, 20]

    json_files = [f for f in os.listdir(input_path) if f.endswith(".json")]
    for f in json_files:
        full_path = os.path.join(input_path, f)
        dataset_name = os.path.splitext(f)[0]
        print(f"Processing {f}...")
        process_kcore(full_path, os.path.join(output_path, dataset_name), k_values)
