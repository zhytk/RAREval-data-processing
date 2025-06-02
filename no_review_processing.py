# no_review_processing.py

import os
import json
import shutil

# Target datasets to process
datasets = ["reviews_Amazon_Instant_Video", "reviews_Digital_Music",
            "reviews_Musical_Instruments", "reviews_Video_Games"]

# Files to remove review text from
files_to_process = ["train_raw.jsonl", "valid_raw.jsonl", "test_raw.jsonl"]

def process_no_reviews(data_path, save_path):
    """
    Removes the 'reviewText' field from each JSONL entry.
    """
    with open(data_path, 'r', encoding='utf-8') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            data.pop("reviewText", None)  # Remove review text if present
            f_out.write(json.dumps(data) + '\n')

def process_dataset(input_root, output_root):
    """
    Processes all datasets under input_root and saves no-review versions to output_root.
    """
    os.makedirs(output_root, exist_ok=True)

    for dataset in datasets:
        dataset_path = os.path.join(input_root, dataset)
        save_dataset_path = os.path.join(output_root, dataset)
        os.makedirs(save_dataset_path, exist_ok=True)

        for file_name in files_to_process:
            input_file = os.path.join(dataset_path, file_name)
            output_file = os.path.join(save_dataset_path, file_name)
            print(f"Processing {input_file}")
            process_no_reviews(input_file, output_file)

        # Copy item/user files without modification
        for aux_file in ["item.jsonl", "user.jsonl"]:
            src = os.path.join(dataset_path, aux_file)
            dst = os.path.join(save_dataset_path, aux_file)
            if os.path.exists(src):
                shutil.copy(src, dst)

    print("✅ No-Review dataset generation complete.")

if __name__ == "__main__":
    process_dataset("data/true-data", "data/no-review")
