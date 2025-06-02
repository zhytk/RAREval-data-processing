# reduction_processing.py

import os
import json
import random
import shutil
import pandas as pd

# Default random seed for reproducibility
RANDOM_SEED = 297370238
random.seed(RANDOM_SEED)

# Target datasets and files
datasets = ["reviews_Amazon_Instant_Video", "reviews_Digital_Music",
            "reviews_Musical_Instruments", "reviews_Video_Games"]

files_to_process = ["train_raw.jsonl", "valid_raw.jsonl", "test_raw.jsonl"]

def process_reduce_reviews(data_path, save_path, user_df, item_df, reduction_degree):
    """
    Randomly removes a percentage of reviewText from user and item review sets.
    Only reviews in train_raw.jsonl are processed.
    """
    with open(data_path, 'r', encoding='utf-8') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
        lines = list(f_in)

        for line in lines:
            data = json.loads(line)
            userID = data["userID"]
            itemID = data["itemID"]
            reviewID = data.get("reviewID")

            # Extract review ID lists
            user_review_list = list(map(int, user_df["reviewIDs"].iloc[int(userID)].split(",")))
            item_review_list = list(map(int, item_df["reviewIDs"].iloc[int(itemID)].split(",")))

            # Sample IDs to remove
            num_to_remove_user = int(len(user_review_list) * reduction_degree)
            num_to_remove_item = int(len(item_review_list) * reduction_degree)
            user_remove_ids = set(random.sample(user_review_list, num_to_remove_user))
            item_remove_ids = set(random.sample(item_review_list, num_to_remove_item))

            # Remove reviewText based on sampled IDs
            if "reviewText" in data:
                if reviewID in user_remove_ids or reviewID in item_remove_ids:
                    del data["reviewText"]

            f_out.write(json.dumps(data) + '\n')

def process_dataset(input_root, output_root, reduction_degree):
    """
    Processes datasets to generate reduced-review versions.
    """
    for dataset in datasets:
        dataset_path = os.path.join(input_root, dataset)
        save_dataset_path = os.path.join(output_root, dataset)
        os.makedirs(save_dataset_path, exist_ok=True)

        # Load user/item review index files
        user_df = pd.read_json(os.path.join(dataset_path, "user.jsonl"), lines=True)
        item_df = pd.read_json(os.path.join(dataset_path, "item.jsonl"), lines=True)

        for file_name in files_to_process:
            input_file = os.path.join(dataset_path, file_name)
            output_file = os.path.join(save_dataset_path, file_name)

            if file_name == "train_raw.jsonl":
                print(f"Processing reduction {int(reduction_degree*100)}%: {input_file}")
                process_reduce_reviews(input_file, output_file, user_df, item_df, reduction_degree)
            else:
                shutil.copy(input_file, output_file)

        # Copy user/item metadata
        for aux in ["user.jsonl", "item.jsonl"]:
            src = os.path.join(dataset_path, aux)
            dst = os.path.join(save_dataset_path, aux)
            if os.path.exists(src):
                shutil.copy(src, dst)

    print(f"✅ Completed processing for {int(reduction_degree*100)}% reduction.\n")

if __name__ == "__main__":
    for rd in [0.25, 0.50, 0.75]:
        output_dir = f"data/reduction/{int(rd * 100)}_percent"
        process_dataset("data/true-data", output_dir, rd)
