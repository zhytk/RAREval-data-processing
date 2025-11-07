import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, logging
# from llama_cpp import Llama
from tqdm import tqdm
import argparse
import torch

LLM = "llama3.2:1b"
'''
MODEL = {
    "llama3:8b": ("bartowski/Meta-Llama-3-8B-Instruct-GGUF", "llama-3"),
    "qwen2:7b": ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen"),
}
'''

MODEL = {
    "llama3:8b": ("meta-llama/meta-llama-3-8b-instruct", "llama-3"),
    "qwen2:7b": ("Qwen/Qwen2-7B-Instruct", "qwen"),
    "llama3.2:1b": ("meta-llama/Llama-3.2-1B-Instruct", "llama-3"),
    "qwen2.5:0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "qwen"),
    "qwen2.5:1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
}

DATASET_DIR = "musical_instruments_5_2014_test" # 保存 preprocessed 数据集的地方

MAX_REVIEWS = 15
MAX_REVIEW_TOKENS = 384
LOG_FREQ = 1<<32
INDEX = 0
NUM_SHARDS = 1
SHOW_RATING = False

# Parameters that cannot be set from command line yet
RANDOM_SEED_LLM = 217116771

def truncate_text(text, max_tokens):
    # return llm.detokenize(tokenizer(text.encode("utf-8"))[:max_tokens]).decode(errors='replace')
    return tokenizer.decode(tokenizer(text, max_length=max_tokens, 
                                      truncation=True, 
                                      add_special_tokens=False)["input_ids"])

def summarize_reviews(text, mode, id):
    system_prompt = {"user": "You are a helpful assistant who analyzes user reviews to extract out user preferences.",
                     "item": "You are a helpful assistant who analyzes user reviews of items to extract key "
                     "characteristics of the item."}
    instruction = {
        "user": (f"Analyze the following user reviews by user {id} and summarize the preferences and key aspects that the "
                 "user values most. Highlight any recurring themes or specific features that influence their ratings "
                 "positively or negatively.\n"),
        "item": (f"Analyze the following user reviews of item {id}. Summarize the key characteristics, features, "
                 "and qualities that users frequently mention. Highlight both positive and negative aspects and "
                 "how they impact the overall perception of the item.\n"),
    }
    #print(system_prompt[mode])
    #print(instruction[mode] + text)
    #assert False

    # Taken from https://huggingface.co/docs/transformers/en/llm_tutorial#wrong-prompt
    messages = [
        {"role": "system", "content": system_prompt[mode]},
        {"role": "user", "content": instruction[mode] + text }
    ]
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = llm.generate(model_inputs, do_sample=True, max_new_tokens=2048)
    output_text = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    #print(output_text)

    # Second-stage chain-of-thought
    messages = [
        {"role": "system", "content": system_prompt[mode]},
        {"role": "assistant", "content": output_text },
        {"role": "user", "content": "Now, summarize what you just wrote in one sentence." },
    ]
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = llm.generate(model_inputs, do_sample=True, max_new_tokens=100)
    output_text = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

    #print(output_text)
    #assert False
    return output_text

def llm_preprocess(train_df, df, mode):
    review_userIDs = train_df["userID"]
    review_itemIDs = train_df["itemID"]
    rating_column = train_df["rating"]
    review_column = train_df["reviewText"]
    df["reviewSummary"] = ""
    n = df.shape[0]
    reviewIDs_list = df["reviewIDs"]

    query_text_length = [0 for i in range(n)]
    start_index = INDEX * n // NUM_SHARDS
    end_index = (INDEX+1) * n // NUM_SHARDS
    for i in tqdm(range(start_index, end_index), smoothing=0.03):
        if mode == "user":
            assert df["userID"].iloc[i] == i
        elif mode == "item":
            assert df["itemID"].iloc[i] == i

        if reviewIDs_list.iloc[i] == "":
            reviewIDs = []
        else:
            reviewIDs = list(map(int, reviewIDs_list.iloc[i].split(",")))
        num_original_reviews = len(reviewIDs)
        if len(reviewIDs) > MAX_REVIEWS:
            reviewIDs = reviewIDs[:MAX_REVIEWS]
        '''
        if num_original_reviews > MAX_REVIEWS:
            print(reviewIDs)
            assert False
        '''

        concat_text = []
        count_nonempty_reviews = 0
        for j in reviewIDs:
            review_text = truncate_text(review_column.iloc[j], MAX_REVIEW_TOKENS)
            otherID = {
                "user": review_itemIDs.iloc[j],
                "item": review_userIDs.iloc[j],
            }

            if review_text == "":
                continue

            if SHOW_RATING:
                review_header = {
                    "user": "Review for item %d (rating %d out of 5)" % (otherID["user"], rating_column.iloc[j]),
                    "item": "Review by user %d (rating %d out of 5)" % (otherID["item"], rating_column.iloc[j]),
                }
            else:
                review_header = {
                    "user": "Review for item %d" % (otherID["user"]),
                    "item": "Review by user %d" % (otherID["item"]),
                }

            concat_text.append(review_header[mode] + ": " + review_text)

            if review_text != "":
                count_nonempty_reviews += 1

        query_text = '\n'.join(concat_text)
        query_text_length[i] = len(query_text)
        # print(query_text)
        # assert False
        df.at[df.index[i], "reviewSummary"] = "" if count_nonempty_reviews == 0 else summarize_reviews(query_text, mode, i)
        # print(df.at[df.index[i], "reviewSummary"])
        # assert False

        if (i+1) % LOG_FREQ == 0:
            num_tokens = len(llm.tokenize(query_text.encode('utf-8')))
            print("%s %d of %d: %d tokens, %s review(s)" % 
                  (mode, i+1, n, num_tokens, len(reviewIDs)))
    
    # https://www.geeksforgeeks.org/ways-to-filter-pandas-dataframe-by-column-values/
    if mode == "user":
        df = df[(df["userID"] >= start_index) & (df["userID"] < end_index)]
    elif mode == "item":
        df = df[(df["itemID"] >= start_index) & (df["itemID"] < end_index)]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--llm', default=LLM, dest='llm', type=str)
    parser.add_argument('--max_reviews', default=MAX_REVIEWS, dest='max_reviews', type=int)
    parser.add_argument('--max_review_tokens', default=MAX_REVIEW_TOKENS, dest='max_review_tokens', type=int)
    parser.add_argument('--log_freq', default=LOG_FREQ, dest='log_freq', type=int)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, dest='dataset_dir', type=str)
    parser.add_argument('--index', default=INDEX, dest='index', type=int)
    parser.add_argument('--num_shards', default=NUM_SHARDS, dest='num_shards', type=int)
    args = parser.parse_args()
    print(args)

    if args.llm == "qwen2:7b" and ("qwen" not in args.dataset_dir):
        raise Exception("Did you forget to add in qwen?")
    
    if args.llm == "llama3:8b" and ("qwen" in args.dataset_dir):
        raise Exception("Did you mix up the dataset directory naming?")

    #args.dataset_dir = "data/" + args.dataset_dir

    LLM = args.llm
    MAX_REVIEWS = args.max_reviews
    MAX_REVIEW_TOKENS = args.max_review_tokens
    LOG_FREQ = args.log_freq
    DATASET_DIR = args.dataset_dir
    INDEX = args.index
    NUM_SHARDS = args.num_shards

    '''
    llm = Llama.from_pretrained(
        repo_id=MODEL[LLM][0],
        filename="*-fp16.gguf",
        n_gpu_layers=-1, # For GPU acceleration (if supported)
        embedding=False,
        chat_format=MODEL[LLM][1],
        verbose=False,
        n_ctx=8192,
        seed=RANDOM_SEED_LLM,
        flash_attn=True, # If CUDA supports, then this may help speed up the program
    )'''
    set_seed(RANDOM_SEED_LLM)
    llm = AutoModelForCausalLM.from_pretrained(MODEL[LLM][0], torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL[LLM][0], use_fast=True)

    # hide warning appearing every iteration
    # https://stackoverflow.com/questions/73221277/python-hugging-face-warning 
    logging.set_verbosity_error()

    FILENAME_SUFFIX = ""
    if NUM_SHARDS != 1:
        FILENAME_SUFFIX = "_%d_of_%d" % (INDEX, NUM_SHARDS)

    train_df = pd.read_json(DATASET_DIR + "/train_raw.jsonl", orient="records", lines=True)
    user_df = pd.read_json(DATASET_DIR + "/user.jsonl", orient="records", lines=True)
    item_df = pd.read_json(DATASET_DIR + "/item.jsonl", orient="records", lines=True)

    # item_df = llm_preprocess(train_df, item_df, "item")

    # https://stackoverflow.com/questions/26837998/pandas-replace-nan-with-blank-empty-string
    train_df["reviewText"] = train_df["reviewText"].fillna("")

    user_df = llm_preprocess(train_df, user_df, "user")
    # user_df = user_df[user_df["reviewSummary"] != ""]
    user_df.to_json(DATASET_DIR + "/user_preferences_short" + FILENAME_SUFFIX + ".jsonl", orient="records", lines=True)

    item_df = llm_preprocess(train_df, item_df, "item")
    # item_df = item_df[item_df["reviewSummary"] != ""]
    item_df.to_json(DATASET_DIR + "/item_descriptions_short" + FILENAME_SUFFIX + ".jsonl", orient="records", lines=True)
