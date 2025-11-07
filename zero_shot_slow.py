import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, pipeline, set_seed, logging
from tqdm import tqdm
import argparse
import torch

# Taken from https://github.com/noamgat/lm-format-enforcer
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

LLM = "llama3.2:3b"
MODEL = {
    "llama3:8b": "meta-llama/meta-llama-3-8b-instruct",
    "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct",
}
DATASET_DIR = "data/true-data/reviews_Musical_Instruments" # 保存 preprocessed 数据集的地方

# 可以在cmd/shell更改
MAX_USER_REVIEWS = 10
MAX_ITEM_REVIEWS = 10
MAX_REVIEW_TOKENS = 384

SHOW_RATINGS = True
SHOW_REVIEWS = True
BATCH_SIZE = 3

# Parameters that cannot be set from command line yet
RANDOM_SEED_LLM = 656563940

def truncate_text(text, max_tokens):
    # return llm.detokenize(tokenizer(text.encode("utf-8"))[:max_tokens]).decode(errors='replace')
    return tokenizer.decode(tokenizer(text, max_length=max_tokens, 
                                      truncation=True, 
                                      add_special_tokens=False)["input_ids"])

# Modified from https://github.com/noamgat/lm-format-enforcer
class Rating(BaseModel):
    rating: float

tokenizer = None
llm = None

def get_prompt_from_two_parts(system_prompt, query_text):
    # print(system_prompt, query_text)
    messages = [
        {"role": "system", "content": system_prompt },
        {"role": "user", "content": query_text }
    ]
    # Modified from https://github.com/noamgat/lm-format-enforcer
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
    prompt = tokenizer.decode(prompt_tokens)
    # print(prompt)
    return prompt

def get_rating(prompt):
    # Taken from https://github.com/noamgat/lm-format-enforcer
    parser = JsonSchemaParser(Rating.model_json_schema())
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    
    # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
    # (suppress warning)
    # https://huggingface.co/docs/transformers/v4.45.2/pipeline_tutorial (batch size)
    responses = llm(prompt, prefix_allowed_tokens_fn=prefix_function, max_new_tokens=100,
                   pad_token_id=tokenizer.eos_token_id, batch_size=BATCH_SIZE)
    
    predictions = []
    for i in range(len(responses)):
        response = responses[i]
        output_json = response[0]['generated_text'][len(prompt[i]):]

        # print(Rating.model_json_schema(), output_json)
        # assert False
        # print(output_json)
        predictions.append(max(min(json.loads(output_json)["rating"], 5), 1))
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_user_reviews', default=MAX_USER_REVIEWS, dest='max_user_reviews', type=int)
    parser.add_argument('--max_item_reviews', default=MAX_ITEM_REVIEWS, dest='max_item_reviews', type=int)
    parser.add_argument('--max_review_tokens', default=MAX_REVIEW_TOKENS, dest='max_review_tokens', type=int)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, dest='dataset_dir', type=str)
    parser.add_argument('--show_ratings', default=SHOW_RATINGS, dest='show_ratings',
                        type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--show_reviews', default=SHOW_REVIEWS, dest='show_reviews',
                        type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--llm', default=LLM, dest='llm', type=str)
    parser.add_argument('--batch_size', default=BATCH_SIZE, dest='batch_size', type=int)
    args = parser.parse_args()
    print(args)

    DATASET_DIR_WITHOUT_PREFIX = args.dataset_dir.replace("/", "_") # Bug fix with Gemini Google Search AI ("how to replace / with _")

    #args.dataset_dir = "data/" + args.dataset_dir

    MAX_USER_REVIEWS = args.max_user_reviews
    MAX_ITEM_REVIEWS = args.max_item_reviews
    MAX_REVIEW_TOKENS = args.max_review_tokens
    DATASET_DIR = args.dataset_dir
    LLM = args.llm
    SHOW_REVIEWS = args.show_reviews
    SHOW_RATINGS = args.show_ratings
    BATCH_SIZE = args.batch_size

    set_seed(RANDOM_SEED_LLM)
    # Pipeline code modified from https://github.com/noamgat/lm-format-enforcer (declare pipeline)
    # https://stackoverflow.com/questions/74748116/huggingface-automodelforcasuallm-decoder-only-architecture-warning-even-after
    # (padding side left)
    llm = pipeline('text-generation', model=MODEL[LLM], torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL[LLM], use_fast=True, device_map="balanced")
    llm.tokenizer.pad_token_id = tokenizer.eos_token_id
    llm.tokenizer.padding_side = "left"

    user_df = pd.read_json(DATASET_DIR + "/user.jsonl", lines=True)
    item_df = pd.read_json(DATASET_DIR + "/item.jsonl", lines=True)
    train_df = pd.read_json(DATASET_DIR + "/train_raw.jsonl", lines=True)
    test_df = pd.read_json(DATASET_DIR + "/test_raw.jsonl", lines=True)
    
    userID_column = test_df["userID"]
    itemID_column = test_df["itemID"]

    n_rows = test_df.shape[0]
    n_users = user_df.shape[0]
    n_items = item_df.shape[0]

    N_SAMPLES = n_rows
    pred_rating = np.zeros(N_SAMPLES)
    actual_rating = np.zeros(N_SAMPLES)
    abs_error = np.zeros(N_SAMPLES)
    squared_error = np.zeros(N_SAMPLES)
    query_text_length = [0 for i in range(N_SAMPLES)]
    confusion_matrix = np.zeros((4, 5), dtype=np.int64)
    total_squared_error = 0

    def get_review_query(userID, itemID):
        concat_query_text = []
        user_review_list = list(map(int, user_df["reviewIDs"].iloc[userID].split(",")))
        item_review_list = list(map(int, item_df["reviewIDs"].iloc[itemID].split(",")))

        user_review_list = user_review_list[:min(MAX_USER_REVIEWS, len(user_review_list))]
        item_review_list = item_review_list[:min(MAX_ITEM_REVIEWS, len(item_review_list))]
        
        # print(user_review_list, item_review_list)

        concat_query_text.append("Here are the reviews written by user %d:" % userID)
        for rowID in user_review_list:
            other_itemID = train_df["itemID"].iloc[rowID]
            rating = train_df["rating"].iloc[rowID]
            review_text = truncate_text(train_df["reviewText"].iloc[rowID], MAX_REVIEW_TOKENS)
            rating_text = (" (rating %d out of 5)" % rating) if SHOW_RATINGS else ""
            review_format_text = ": %s" % review_text if (SHOW_REVIEWS and review_text != "") else ""
            review_line = "Review for item %d%s%s" % (other_itemID, rating_text, review_format_text)
            concat_query_text.append(review_line)

        concat_query_text.append('')
        concat_query_text.append("Here are the reviews written for item %d:" % itemID)
        for rowID in item_review_list:
            other_userID = train_df["userID"].iloc[rowID]
            rating = train_df["rating"].iloc[rowID]
            review_text = truncate_text(train_df["reviewText"].iloc[rowID], MAX_REVIEW_TOKENS)
            rating_text = (" (rating %d out of 5)" % rating) if SHOW_RATINGS else ""
            review_format_text = ": %s" % review_text if (SHOW_REVIEWS and review_text != "") else ""
            review_line = "Review by user %d%s%s" % (other_userID, rating_text, review_format_text)
            concat_query_text.append(review_line)
        
        return '\n'.join(concat_query_text)

    def get_prompt(i: int):
        userID = userID_column.iloc[i]
        itemID = itemID_column.iloc[i]

        question_text = ("User %d now bought item %d. "
                        "Give your rating prediction in the JSON schema: {\"rating\": {\"type\": \"number\"}}. "
                        "Do not give reasoning.") % (userID, itemID)

        format_template_text = " (rating x out of 5): [review]."
        if not SHOW_REVIEWS and SHOW_RATINGS:
            format_template_text = " (rating x out of 5)."
        elif SHOW_REVIEWS and not SHOW_RATINGS:
            format_template_text = ": [review]."
        elif not SHOW_REVIEWS and not SHOW_RATINGS:
            format_template_text = "."

        concat_text = [("You're required to predict user ratings for item recommendations. The rating ranges from 1.0 to 5.0. "
                        "You're given the user's past review and rating history of items in the format: "
                        "Review for item xxx%s "
                        "You're given the item's past user-generated review and rating history in the format: "
                        "Review by user xxx%s") % (format_template_text, format_template_text), '']
        
        concat_text.append(get_review_query(userID, itemID))
        concat_text.append('')
        concat_text.append(question_text)

        system_prompt = concat_text[0]
        query_text = '\n'.join(concat_text[2:])
        return get_prompt_from_two_parts(system_prompt, query_text)

    class PromptDataset(torch.utils.data.Dataset):
        def __init__(self):
            pass

        def __getitem__(self, idx):
            return get_prompt(idx)
        
        def __len__(self):
            return N_SAMPLES

    prompt_dataset = PromptDataset()
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE), smoothing=0.03):
        start = i
        end = min(N_SAMPLES, i+BATCH_SIZE)
        prompts = []
        for j in range(start, end):
            prompts.append(prompt_dataset[j])

        pred_ratings = get_rating(prompts)
        for j in range(start, end):
            pred_rating[j] = pred_ratings[j-start]
        # pred_rating[i] = get_rating(prompt_dataset[i])
        # print(prompt_dataset[i])
        # print(pred_rating[i])
        # assert False

    for i in range(N_SAMPLES):
        actual_rating[i] = test_df["rating"].iloc[i]
        abs_error[i] = abs(pred_rating[i] - actual_rating[i])
        squared_error[i] = (pred_rating[i] - actual_rating[i]) ** 2
        total_squared_error += squared_error[i]

        if pred_rating[i] == 5:
            pred_row = 3
        elif pred_rating[i] > 4:
            pred_row = 2
        elif pred_rating[i] == 4:
            pred_row = 1
        elif pred_rating[i] < 4:
            pred_row = 0
        else:
            assert False

        # print(pred_rating[i], actual_rating[i])

        confusion_matrix[pred_row][int(actual_rating[i]-1)] += 1

    alpha = 0.05 # 95% confidence interval
    mse = np.mean(squared_error)
    mae = np.mean(abs_error)
    std = np.std(squared_error, ddof=1)
    std_error = std / np.sqrt(N_SAMPLES) * np.sqrt((n_rows - N_SAMPLES) / (n_rows - 1))
    print("sample mse is %.4f, mae is %.4f, std of squared error is %.4f, standard error is %.4f" % 
                (mse, mae, std, std_error))
    print(confusion_matrix)
    print(np.array_str(confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True), precision=3))
    print(np.array_str(confusion_matrix / np.sum(confusion_matrix, axis=0, keepdims=True), precision=3))

    # Compute training frequency
    # Required for evaluation of MSE by item training frequency
    train_user_ids = train_df["userID"].tolist()
    train_item_ids = train_df["itemID"].tolist()
    test_user_ids = test_df["userID"].tolist()
    test_item_ids = test_df["itemID"].tolist()

    train_user_freq_np = np.unique(train_user_ids, return_counts=True)
    train_item_freq_np = np.unique(train_item_ids, return_counts=True)
    train_user_freq = {}
    train_item_freq = {}
    max_train_user_freq = np.max(train_user_freq_np[1])
    max_train_item_freq = np.max(train_item_freq_np[1])
    print("Maximum item training frequency per epoch: " + str(max_train_user_freq))
    print("Maximum user training frequency per epoch: " + str(max_train_item_freq))
    for i in range(train_user_freq_np[0].shape[0]):
        train_user_freq[train_user_freq_np[0][i]] = train_user_freq_np[1][i]

    for i in range(train_item_freq_np[0].shape[0]):
        train_item_freq[train_item_freq_np[0][i]] = train_item_freq_np[1][i]

    # print(train_user_freq)
    # print(train_item_freq)

    train_user_freq_count = np.zeros(max_train_user_freq+1)
    train_item_freq_count = np.zeros(max_train_item_freq+1)
    
    train_user_freq_list = np.zeros(N_SAMPLES)
    train_item_freq_list = np.zeros(N_SAMPLES)
    user_rating_count_list = np.zeros(N_SAMPLES)
    item_rating_count_list = np.zeros(N_SAMPLES)
    for i in range(N_SAMPLES):
        user_freq = train_user_freq[test_user_ids[i]]
        item_freq = train_item_freq[test_item_ids[i]]
        train_user_freq_count[user_freq] += 1
        train_item_freq_count[item_freq] += 1
        train_user_freq_list[i] = user_freq
        train_item_freq_list[i] = item_freq

        # result = test_dataset.get_attr(i)
        # print(result)
        # assert False
        # user_rating_count_list[i] = result["user_rating_count"]
        # item_rating_count_list[i] = result["item_rating_count"]

    # print(pred_rating, actual_rating)
    results_df = pd.DataFrame({
        "train_user_freq": train_user_freq_list, 
        "train_item_freq": train_item_freq_list, 
        # "user_rating_count": user_rating_count_list,
        # "item_rating_count": item_rating_count_list,
        "predicted": pred_rating,
        "actual": actual_rating})

    results_df.to_json("zero_shot_%s_%s_results.jsonl" % (DATASET_DIR_WITHOUT_PREFIX, LLM.replace(":", "_")), 
                       orient="records", lines=True)
