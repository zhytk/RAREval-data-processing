import json
import numpy as np
import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm
import argparse
import random
import  os

LLM = "llama3.2:3b"
MODEL = {
    "llama3:8b": ("bartowski/Meta-Llama-3-8B-Instruct-GGUF", "llama-3", "*fp16.gguf"),
    "llama3.1:8b": ("professorf/Meta-Llama-3-1-8B-Instruct-f16-gguf", "llama-3", "*f16.gguf"),
    "llama3.2:3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "llama-3", "*f16.gguf"),
    "llama3.2:1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "llama-3", "*f16.gguf"),
    "qwen2:7b": ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen", "*fp16.gguf"),
    "qwen2.5:14b": ("bartowski/Qwen2.5-14B-Instruct-GGUF", "qwen", "*f16.gguf"),
    "qwen2.5:7b": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "qwen", "*f16.gguf"),
    "qwen2.5:3b": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "qwen", "*f16.gguf"),
    "qwen2.5:1.5b": ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "qwen", "*f16.gguf"),
    "qwen2.5:0.5b": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "qwen", "*f16.gguf"),
}
DATASET_DIR = "data/true-data/reviews_Musical_Instruments" # 保存 preprocessed 数据集的地方
# 可以在cmd/shell更改
MAX_USER_REVIEWS = 6
MAX_ITEM_REVIEWS = 6
MAX_REVIEW_TOKENS = 384

SHOW_RATINGS = True
SHOW_REVIEWS =True

# Parameters that cannot be set from command line yet
RANDOM_SEED_LLM = 656563940
RANDOM_SEED = 189403968
MAX_EXAMPLE_USER_REVIEWS = 2
MAX_EXAMPLE_ITEM_REVIEWS = 2
MAX_EXAMPLE_REVIEW_TOKENS = 256
N_EXAMPLES = 3

def truncate_text(text, max_tokens):
    return llm.detokenize(llm.tokenize(text.encode("utf-8"))[:max_tokens]).decode(errors='replace')

def get_rating(system_prompt, query_text):
    # print(system_prompt, query_text)
    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": query_text }
        ],
        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"rating": {"type": "number", "minimum": 1, "maximum": 5}},
                "required": ["rating"],
            },
        },
    )
    output_json = response["choices"][0]["message"]["content"]
    return max(min(json.loads(output_json)["rating"], 5), 1)

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
    args = parser.parse_args()
    print(args)

    DATASET_DIR_WITHOUT_PREFIX = args.dataset_dir.replace("/", "_") # Bug fix with Gemini Google Search AI ("how to replace / with _")

    # args.dataset_dir = "data/" + args.dataset_dir

    MAX_USER_REVIEWS = args.max_user_reviews
    MAX_ITEM_REVIEWS = args.max_item_reviews
    MAX_REVIEW_TOKENS = args.max_review_tokens
    DATASET_DIR = args.dataset_dir
    LLM = args.llm
    SHOW_RATINGS = args.show_ratings
    SHOW_REVIEWS = args.show_reviews

    llm = Llama.from_pretrained(
        repo_id=MODEL[LLM][0],
        filename=MODEL[LLM][2],
        n_gpu_layers=-1, # For GPU acceleration (if supported)
        embedding=False,
        chat_format=MODEL[LLM][1],
        verbose=False,
        n_ctx=8192,
        seed=RANDOM_SEED_LLM,
        flash_attn=True, # If CUDA supports, then this may help speed up the program
    )

    random.seed(RANDOM_SEED)

    user_df = pd.read_json(DATASET_DIR + "/user.jsonl", lines=True)
    item_df = pd.read_json(DATASET_DIR + "/item.jsonl", lines=True)
    train_df = pd.read_json(DATASET_DIR + "/train_raw.jsonl", lines=True)
    test_df = pd.read_json(DATASET_DIR + "/test_raw.jsonl", lines=True)

    # https://stackoverflow.com/questions/26837998/pandas-replace-nan-with-blank-empty-string
    train_df["reviewText"] = train_df["reviewText"].fillna("")

    userID_column = test_df["userID"]
    itemID_column = test_df["itemID"]

    train_userID_column = []
    train_itemID_column = []
    train_rating_column = []
    for i in range(len(train_df)):
        # Need to exclude one training review, hence need to check for presence of at least two user reviews and two item reviews
        userID = train_df["userID"].iloc[i]
        itemID = train_df["itemID"].iloc[i]
        rating = train_df["rating"].iloc[i]
        user_review_list = list(map(int, user_df["reviewIDs"].iloc[userID].split(",")))
        item_review_list = list(map(int, item_df["reviewIDs"].iloc[itemID].split(",")))
        if len(user_review_list) >= 2 and len(item_review_list) >= 2:
            train_userID_column.append(userID)
            train_itemID_column.append(itemID)
            train_rating_column.append(rating)

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

    def get_review_query(userID, itemID, is_target, ground_truth_rating):
        concat_query_text = []
        user_review_list = list(map(int, user_df["reviewIDs"].iloc[userID].split(",")))
        item_review_list = list(map(int, item_df["reviewIDs"].iloc[itemID].split(",")))

        max_user_reviews = MAX_EXAMPLE_USER_REVIEWS if not is_target else MAX_USER_REVIEWS
        max_item_reviews = MAX_EXAMPLE_ITEM_REVIEWS if not is_target else MAX_ITEM_REVIEWS

        # user_review_list = user_review_list[:min(max_user_reviews, len(user_review_list))]
        # item_review_list = item_review_list[:min(max_item_reviews, len(item_review_list))]

        max_review_tokens = MAX_REVIEW_TOKENS if is_target else MAX_EXAMPLE_REVIEW_TOKENS

        # print(user_review_list, item_review_list)

        target_text = " target" if is_target else ""

        concat_query_text.append("Q: Below are the %s from%s user_%d for other items:" %
            ("reviews" if SHOW_REVIEWS else "ratings", target_text, userID))
        count = 0
        for rowID in user_review_list:
            other_itemID = train_df["itemID"].iloc[rowID]
            if other_itemID == itemID:
                assert not is_target
                continue
            count += 1
            rating = train_df["rating"].iloc[rowID]
            review_text = truncate_text(train_df["reviewText"].iloc[rowID], max_review_tokens)
            effective_show_reviews = SHOW_REVIEWS and review_text != ""
            if SHOW_RATINGS and effective_show_reviews:
                review_line = "%d, \"%s\"" % (rating, review_text)
            elif SHOW_RATINGS and not effective_show_reviews:
                review_line = "%d" % (rating)
            elif not SHOW_RATINGS and effective_show_reviews:
                review_line = "\"%s\"" % (review_text)
            else:
                assert False
            concat_query_text.append(review_line)
            if count == max_user_reviews:
                break

        concat_query_text.append("Below are the %s from other users for%s item_%d:" %
            ("reviews" if SHOW_REVIEWS else "ratings", target_text, itemID))
        count = 0
        for rowID in item_review_list:
            other_userID = train_df["userID"].iloc[rowID]
            if other_userID == userID:
                assert not is_target
                continue
            count += 1
            rating = train_df["rating"].iloc[rowID]
            review_text = truncate_text(train_df["reviewText"].iloc[rowID], max_review_tokens)
            effective_show_reviews = SHOW_REVIEWS and review_text != ""
            if SHOW_RATINGS and effective_show_reviews:
                review_line = "%d, \"%s\"" % (rating, review_text)
            elif SHOW_RATINGS and not effective_show_reviews:
                review_line = "%d" % (rating)
            elif not SHOW_RATINGS and effective_show_reviews:
                review_line = "\"%s\"" % (review_text)
            else:
                assert False
            concat_query_text.append(review_line)
            if count == max_item_reviews:
                break

        concat_query_text.append(("What's the rating that%s user_%d will give for%s item_%d?"
                                 " Give a single number without giving reasoning.") % (target_text, userID, target_text, itemID))
        concat_query_text.append("A: " if is_target else "A: %.1f" % (float(ground_truth_rating)))

        return '\n'.join(concat_query_text)

    for i in tqdm(range(N_SAMPLES), smoothing=0.03):
        userID = userID_column.iloc[i]
        itemID = itemID_column.iloc[i]

        question_text = ("What's the rating that the target user_%d will give for the target item_%d?"
                         " Provide the rating prediction and do not give reasoning.") % (userID, itemID)

        prompt_format_text = {
            False: {
                False: None,
                True: "You're given a list of user-generated reviews.",
            },
            True: {
                False: "You're given a list of ratings, one rating per line.",
                True: "You're given the rating and the corresponding user-generated review in the format: Rating, Review.",
            }
        }

        concat_text = [("You're required to predict user ratings for item recommendations."
                        " The rating ranges from 1.0 to 5.0. %s" % prompt_format_text[SHOW_RATINGS][SHOW_REVIEWS]), '']

        example_indices = set()
        while len(example_indices) < N_EXAMPLES:
            index = random.randint(0, len(train_userID_column)-1)
            example_itemID = train_itemID_column[index]
            example_userID = train_userID_column[index]
            if example_itemID != itemID and example_userID != userID:
                example_indices.add(index)

        for index in example_indices:
            example_itemID = train_itemID_column[index]
            example_userID = train_userID_column[index]
            example_rating = train_rating_column[index]
            concat_text.append(get_review_query(example_userID, example_itemID, False, example_rating))
            concat_text.append('')

        concat_text.append(get_review_query(userID, itemID, True, -1))
        # concat_text.append('')
        # concat_text.append(question_text)

        system_prompt = concat_text[0]
        query_text = '\n'.join(concat_text[2:])
        query_text_length[i] = len(query_text)

        pred_rating[i] = get_rating(system_prompt, query_text)

        # print("System prompt:", system_prompt)
        # print("Query text:", query_text)
        # assert False

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
    print("Maxmimum item training frequency per epoch: " + str(max_train_user_freq))
    print("Maxmimum user training frequency per epoch: " + str(max_train_item_freq))
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

    results_df = pd.DataFrame({
        "train_user_freq": train_user_freq_list,
        "train_item_freq": train_item_freq_list,
        # "user_rating_count": user_rating_count_list,
        # "item_rating_count": item_rating_count_list,
        "predicted": pred_rating,
        "actual": actual_rating})

    results_df.to_json("few_shot_%s_%s_results.jsonl" % (DATASET_DIR_WITHOUT_PREFIX, LLM.replace(":", "_")),
                    orient="records", lines=True)
