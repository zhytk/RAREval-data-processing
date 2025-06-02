# distortion_processing.py

import os
import json
import random
import shutil
import pandas as pd

RANDOM_SEED = 297370238
random.seed(RANDOM_SEED)

datasets = ["reviews_Amazon_Instant_Video", "reviews_Digital_Music",
            "reviews_Musical_Instruments", "reviews_Video_Games"]

files_to_process = ["train_raw.jsonl", "valid_raw.jsonl", "test_raw.jsonl"]

def process_distort_reviews(data_path, save_path, user_df, item_df, distortion_ratio):
    """
    Randomly distorts a portion of reviewText fields in the training set
    by replacing them with reviews from other samples (review shuffling).
    """
    with open(data_path, 'r', encoding='utf-8') as f_in:
        lines = list(f_in)
    
    review_pool = []
    
    # Step 1: Build review pool from existing reviewText entries
    for line in lines:
        data = json.loads(line)
        if "reviewText" in data:
            review_pool.append({
                "userID": data["userID"],
                "itemID": data["itemID"],
                "reviewText": data["reviewText"]
            })

    # Step 2: Shuffle reviewText in selected samples
    num_to_distort = int(len(review_pool) * distortion_ratio)
    selected_reviews = random.sample(review_pool, num_to_distort)
    distorted_reviews = selected_reviews.copy()
    random.shuffle(distorted_reviews)

    # Apply distortion
    for i in range(num_to_distort):
        selected_reviews[i]["reviewText"] = distorted_reviews[i]["reviewText"]

    # Step 3: Rewrite dataset with distorted reviews
    with open(save_path, 'w', encoding='utf-8') as f_out:
        for line in lines:
            data = json.loads(line)
            uid = data["userID"]
            iid = data["itemID"]

            for r in selected_reviews:
                if r["userID"] == uid and r["itemID"] == iid:
                    data["reviewText"] = r["reviewText"]
                    break

            f_out.write(json.dumps(data) + '\n')

def process_dataset(input_root, output_root, distortion_ratio):
    for dataset in datasets:
        dataset_path = os.path.join(input_root, dataset)
        save_dataset_path = os.path.join(output_root, dataset)
        os.makedirs(save_dataset_path, exist_ok=True)

        # Load user/item info (optional for consistency)
        user_df = pd.read_json(os.path.join(dataset_path, "user.jsonl"), lines=True)
        item_df = pd.read_json(os.path.join(dataset_path, "item.jsonl"), lines=True)

        for file_name in files_to_process:
            input_file = os.path.join(dataset_path, file_name)
            output_file = os.path.join(save_dataset_path, file_name)

            if file_name == "train_raw.jsonl":
                print(f"Distorting {int(distortion_ratio*100)}% of reviews in {input_file}")
                process_distort_reviews(input_file, output_file, user_df, item_df, distortion_ratio)
            else:
                shutil.copy(input_file, output_file)

        for aux in ["user.jsonl", "item.jsonl", "5-core_full.jsonl"]:
            src = os.path.join(dataset_path, aux)
            dst = os.path.join(save_dataset_path, aux)
            if os.path.exists(src):
                shutil.copy(src, dst)

    print(f"✅ Completed {int(distortion_ratio*100)}% distortion\n")

if __name__ == "__main__":
    for dr in [0.25, 0.50, 0.75, 1.00]:
        out_dir = f"data/distortion/{int(dr * 100)}_percent"
        process_dataset("data/true-data", out_dir, dr)
