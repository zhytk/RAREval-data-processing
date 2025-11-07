# Code mostly taken from https://huggingface.co/docs/transformers/en/training
# and https://github.com/huggingface/peft
# Code for model quantization taken from https://huggingface.co/blog/4bit-transformers-bitsandbytes
# from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
# from transformers import BitsAndBytesConfig
# from transformers.utils import logging
from torch.utils.data import DataLoader
# from peft import get_peft_config
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict, load_dataset
from scipy.special import softmax
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
# import pandas as pd
import evaluate
import torch
import os
import argparse
import sys
import contextlib
import pandas as pd
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# logging.set_verbosity(logging.ERROR)

MODEL_DICT = {
    "llama3:8b": "meta-llama/meta-llama-3-8b-instruct",
    "llama3:8b": "meta-llama/meta-llama-3-8b-instruct",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    # can be used when preprocessed with Qwen2:7B but not Llama3:8B due to restrictions over Meta Llama license
    "qwen2:0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2:1.5b": "Qwen/Qwen2-1.5B-Instruct", 
    "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
    "qwen2:72b": "Qwen/Qwen2-72B-Instruct",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
}

LLM = "llama3.2:1b"
# DATASET_DIR = "./musical_instruments_5_2014"
DATASET_DIR = "./musical_instruments_5_2014_test/summarized"
INSTANCE_ID = "testing"
OUTFILE = "testing"
# DATASET_DIR = "./instant_video_5_2014"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# torch.backends.cuda.matmul.allow_tf32 = True
REGRESSION = True

CONTEXT_WINDOW = -1
BATCH_SIZE = 64
LORA_RANK = 64
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
LORA_ALPHA = 64
LOGGING_STEPS = 32
TRAIN_EMBEDDING = True
NUM_EPOCHS = 100
RESUME_FROM_CHECKPOINT = False
SHOW_TOKEN_STATS = True
LORA_DROPOUT = 0
ADD_RATINGS = 1 # 把每一个 review 另外加上 rating
HIDE_REVIEWS = False # 不要给 LLM 任何 review text 来做 sanity check
SUMMARIZED = True # Summarize reviews by default, False to concatenate reviews
LOSS_FUNCTION = "mae" # "mae" for MAE loss, "mse" for MSE loss
RANDOM_SAMPLE = False # change to true to randomly sample different reviews for each epoch during training
EARLY_STOPPING_PATIENCE = 3

# To reduce disk usage, preprocess.py will be integrated into llm_lora.py
MAX_USER_REVIEWS = 15
MAX_ITEM_REVIEWS = 15
MAX_REVIEW_TOKENS = 384

# Cannot be changed via command line
RANDOM_REVIEW_SAMPLE_SEED = 176236834

train_truncated = {}
def truncate_text_cache(train_df, rowID, max_tokens):
    if rowID not in train_truncated:
        # For our use case, max_tokens can be assumed to be a constant
        assert False # The cache should have been built beforehand
        # train_truncated[rowID] = truncate_text(train_df["reviewText"].iloc[rowID], max_tokens)
    return train_truncated[rowID]

def truncate_text(text, max_tokens, tokenizer):
    try:
        tokens = tokenizer(text, max_length=max_tokens, truncation=True, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(tokens)
    except:
        print(text)
        assert False

def build_cache(train_df, max_tokens, tokenizer):
    for rowID in tqdm(range(len(train_df))):
        train_truncated[rowID] = truncate_text(train_df["reviewText"].iloc[rowID], max_tokens, tokenizer)

# This function is used for a major breaking change - in prompt format
# The major breaking change allows for prompts to be evaluated on the fly rather than being stored in the CPU memory.
# However, there is a disadvantage of context window determination.
def get_prompt(user_df, item_df, train_df, test_df, review_idx, is_training=False):
    userID_column = test_df["userID"]
    itemID_column = test_df["itemID"]

    userID = userID_column.iloc[review_idx]
    itemID = itemID_column.iloc[review_idx]

    ground_truth = test_df["rating"].iloc[review_idx]
    data_leakage = ["l;bCeX59JB,by,C", "KrnOJ)b&tgH}0};", "DT;Q=uY0{w8dtkv", "dDx2qlGNeRoCBT;", "b.[(A{fz3@LIFN!", "yn0PUE7x)eZKM@q"]

    concat_text = []

    if user_df["reviewIDs"].iloc[userID] == "":
        user_review_list = []
    else:
        user_review_list = list(map(int, user_df["reviewIDs"].iloc[userID].split(",")))
    
    if item_df["reviewIDs"].iloc[itemID] == "":
        item_review_list = []
    else:
        item_review_list = list(map(int, item_df["reviewIDs"].iloc[itemID].split(",")))

    if is_training and args.random_sample:
        user_review_list = random.sample(user_review_list, min(MAX_USER_REVIEWS, len(user_review_list)))
        item_review_list = random.sample(item_review_list, min(MAX_ITEM_REVIEWS, len(item_review_list)))
    else:
        user_review_list = user_review_list[:min(MAX_USER_REVIEWS, len(user_review_list))]
        item_review_list = item_review_list[:min(MAX_ITEM_REVIEWS, len(item_review_list))]
    '''
    if is_training:
        random.shuffle(user_review_list)
        random.shuffle(item_review_list)
    '''
    
    user_rating_str_without_id = "ratings provided by user%d:" % userID
    user_rating_str_with_id = [user_rating_str_without_id]
    user_review_text_list = []
    user_rating_list = []

    item_rating_str_without_id = "ratings provided for item%d:" % itemID
    item_rating_str_with_id = [item_rating_str_without_id]
    item_review_text_list = []
    item_rating_list = []

    for rowID in user_review_list:
        other_itemID = train_df["itemID"].iloc[rowID]
        other_rating = train_df["rating"].iloc[rowID]
        review_text = truncate_text_cache(train_df, rowID, MAX_REVIEW_TOKENS)
        # review_header = "Review for item %d" % (other_itemID)

        if not HIDE_REVIEWS:
            review_header = {0: "Review: ", 
                1: "Rating %d out of 5: " % (other_rating),
                2: "Review for item %d (rating %s out of 5): " % 
                (other_itemID, str(other_rating) if other_itemID != itemID else "hidden")}
            user_review_text_list.append(review_header[ADD_RATINGS] + review_text)

        if other_itemID == itemID:
            # Skip the rating, otherwise, it will be considered leakage
            pass
        else:
            user_rating_list.append(str(other_rating))
            user_rating_str_with_id.append("item%d:%d" % (other_itemID, other_rating))
             

    for rowID in item_review_list:
        other_userID = train_df["userID"].iloc[rowID]
        other_rating = train_df["rating"].iloc[rowID]
        review_text = truncate_text_cache(train_df, rowID, MAX_REVIEW_TOKENS)
        # review_header = "Review by user %d" % (other_userID)

        if not HIDE_REVIEWS:
            review_header = {0: "Review: ", 
                1: "Rating %d out of 5: " % (other_rating),
                2: "Review by user %d (rating %s out of 5): " % 
                (other_userID, str(other_rating) if other_userID != userID else "hidden")}
            item_review_text_list.append(review_header[ADD_RATINGS] + review_text)

        if other_userID == userID:
            # Skip the rating, otherwise, it will be considered leakage
            pass
        else:
            item_rating_list.append(str(other_rating))
            item_rating_str_with_id.append("user%d:%d" % (other_userID, other_rating))

    user_rating_count = len(user_rating_list)
    item_rating_count = len(item_rating_list)
    user_rating_str_without_id += ",".join(user_rating_list)
    item_rating_str_without_id += ",".join(item_rating_list)
    user_rating_str_with_id = "\n".join(user_rating_str_with_id)
    item_rating_str_with_id = "\n".join(item_rating_str_with_id)
    user_review_str = "\n".join(user_review_text_list)
    item_review_str = "\n".join(item_review_text_list)
    
    if HIDE_REVIEWS and ADD_RATINGS == 0:
        # Short-circuit for review hiding and rating hiding, more efficient use of context window
        concat_text.append("user %d, item %d" % (userID, itemID))
    elif HIDE_REVIEWS and ADD_RATINGS == 1:
        concat_text.append(user_rating_str_without_id)
        concat_text.append(item_rating_str_without_id)
        # concat_text.append(data_leakage[ground_truth])
    elif HIDE_REVIEWS and ADD_RATINGS == 2:
        concat_text.append(user_rating_str_with_id)
        concat_text.append(item_rating_str_with_id)
    elif not HIDE_REVIEWS and ADD_RATINGS == 2 and SUMMARIZED:
        concat_text.append("user preferences:" + user_df["reviewSummary"].iloc[userID])
        concat_text.append("item description:" + item_df["reviewSummary"].iloc[itemID])
        # concat_text.append(data_leakage[ground_truth])
        concat_text.append(user_rating_str_with_id)
        concat_text.append(item_rating_str_with_id)
    elif not HIDE_REVIEWS and ADD_RATINGS == 1 and SUMMARIZED:
        concat_text.append("user preferences:" + user_df["reviewSummary"].iloc[userID])
        concat_text.append("item description:" + item_df["reviewSummary"].iloc[itemID])
        # concat_text.append(data_leakage[ground_truth])
        concat_text.append(user_rating_str_without_id)
        concat_text.append(item_rating_str_without_id)
    elif not HIDE_REVIEWS and ADD_RATINGS == 0 and SUMMARIZED:
        concat_text.append("user preferences:" + user_df["reviewSummary"].iloc[userID])
        concat_text.append("item description:" + item_df["reviewSummary"].iloc[itemID])
    elif not HIDE_REVIEWS and not SUMMARIZED:
        concat_text.append("Reviews written by user %d:" % userID)
        concat_text.append(user_review_str)
        concat_text.append("Reviews written for item %d:" % itemID)
        concat_text.append(item_review_str)
    else:
        # Not implemented yet
        assert False, "Not implemented yet"

    query_text = '\n'.join(concat_text)
    # print(query_text)
    # assert False
    diff_user = len(user_review_list) - user_rating_count
    assert diff_user == 0 or diff_user == 1
    diff_item = len(item_review_list) - item_rating_count
    assert diff_item == 0 or diff_item == 1
    # assert diff_item == 0
    return {"prompt": query_text, "user_rating_count": user_rating_count, "item_rating_count": item_rating_count}
    # return query_text

# Code courtesy of 
# https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
# by user Wolph
@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

def main(args, output_file = None):
    with smart_open(output_file) as sys.stdout:
        random.seed(RANDOM_REVIEW_SAMPLE_SEED)

        MODEL_DIR = MODEL_DICT[args.llm]

        # if "llama3" not in args.llm and "qwen" not in DATASET_DIR:
        #    raise Exception("Meta Llama 3 license violated")

        # Quantization is not used for this project, though it was considered at earlier stages.
        '''
        four_bit_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        eight_bit_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        '''

        device = "cuda"

        args.local_rank = -1
        print("Instance ID:", args.instance_id)
        print("Outfile:", args.outfile)
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            print("Local rank:", args.local_rank)
            args.is_distributed = True
        else:
            print("Not in distributed setting")
            args.is_distributed = False

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
        if args.llm == "tinyllama":
            # TinyLlama does not come with default chat template in Hugging Face and actually uses Llama 2 chat template
            llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
            tokenizer.chat_template = llama2_tokenizer.chat_template

        # Llama 3 model in Hugging Face apparently does not come with pad token.
        if "llama3:" in args.llm:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "llama3." in args.llm:
            # For Llama 3.1 and Llama 3.2
            # tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer("<|eot_id|>")["input_ids"][0]
        elif "qwen2" in args.llm:
            # https://stackoverflow.com/questions/76045897/how-do-i-get-the-padding-token-id-in-huggingface-tokenizer-for-a-pad-token
            tokenizer.pad_token_id = tokenizer("<|endoftext|>")["input_ids"][0]
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        train_df = pd.read_json(DATASET_DIR + "/train_raw.jsonl", orient="records", lines=True)
        valid_df = pd.read_json(DATASET_DIR + "/valid_raw.jsonl", orient="records", lines=True)
        test_df = pd.read_json(DATASET_DIR + "/test_raw.jsonl", orient="records", lines=True)

        # https://stackoverflow.com/questions/26837998/pandas-replace-nan-with-blank-empty-string
        train_df["reviewText"] = train_df["reviewText"].fillna("")

        def add_inst_tokens(review):
            prompt = review
            query = [{"role": "user", "content": prompt}]
            if "qwen2" in args.llm:
                # Qwen2 and Llama3.2 add a default system prompt if there is no system prompt
                query.insert(0, {"role": "system", "content": ""})
            new_prompt = tokenizer.apply_chat_template(
                query, tokenize=False, add_generation_prompt=False)
            # print(new_prompt)

            if "llama3.2" in args.llm:
                # Llama 3.2 appends knowledge cutoff date and today's date if there is no system prompt,
                # perhaps for knowledge of temporal clues. This is not needed for our finetuning, hence
                # we replace it with an empty system prompt.
                idx = new_prompt.find("<|eot_id|>")
                assert idx != -1
                start_idx = idx + len("<|eot_id|>")
                new_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|>" + new_prompt[start_idx:]
            
            # print(query, new_prompt)
            # print(new_prompt)
            # assert False
            return {"prompt": new_prompt}

        def tokenize_function(review):
            prompt = review["prompt"]
            add_special_tokens = True
            if args.context_window == -1:
                return tokenizer(prompt, padding="longest", add_special_tokens=add_special_tokens)
            else:
                return tokenizer(prompt, max_length=args.context_window, 
                                padding="max_length", truncation=True,
                                add_special_tokens=add_special_tokens)

        # Most likely need to define dataset here
        # The new dataset class needs user, item, training and testing datasets
        # Reference: https://towardsdatascience.com/linear-regression-with-hugging-face-3883fe729324
        # Modified to make return dtype bfloat16
        class MakeTorchData(torch.utils.data.Dataset):
        # class MakeTorchData(torch.utils.data.IterableDataset):
            def __init__(self, user_df, item_df, train_df, test_df, is_training=False):
                self.user_df = user_df
                self.item_df = item_df
                self.train_df = train_df
                self.test_df = test_df
                self.labels = list(self.test_df["rating"])
                self.is_training = is_training

            def get_attr(self, idx):
                return get_prompt(self.user_df, self.item_df, self.train_df, self.test_df, idx, is_training=self.is_training)

            def __getitem__(self, idx):
                result = self.get_attr(idx)
                prompt = result["prompt"]
                # print(prompt)
                # print(result)
                prompt = add_inst_tokens(prompt)
                encoding = tokenize_function(prompt)
                item = {k: torch.tensor(v) for k, v in encoding.items()}
                item["labels"] = self.labels[idx]
                # item["prompt"] = prompt["prompt"] # for debugging only
                return item

            def length(self):
                return len(self.labels)
                # return 3 # for testing
            
            def __len__(self):
                return len(self.labels)
                # return 3 # for testing
            
            def generate(self):
                # To make dataset iterable, with random shuffling after every epoch
                '''order = [j for j in range(self.length())]
                while True:
                    random.shuffle(order)
                    for i in range(self.length()):
                        yield self.__getitem__(i)'''
                
                # Return in sequential order
                for i in range(self.length()):
                    yield self.__getitem__(i)
                

            # https://discuss.pytorch.org/t/how-to-use-dataloader-with-iterabledataset/179242
            def __iter__(self):
                return iter(self.generate())

        build_cache(train_df, MAX_REVIEW_TOKENS, tokenizer)
        if not args.summarized:
            # Concatenated
            user_df = pd.read_json(DATASET_DIR + "/user.jsonl", orient="records", lines=True)
            item_df = pd.read_json(DATASET_DIR + "/item.jsonl", orient="records", lines=True)

            train_dataset = MakeTorchData(user_df, item_df, train_df, train_df, is_training=True)
            eval_dataset = MakeTorchData(user_df, item_df, train_df, valid_df)
            test_dataset = MakeTorchData(user_df, item_df, train_df, test_df)
        else:
            # Summarized
            user_preferences_df = pd.read_json(DATASET_DIR + "/user_preferences_short.jsonl", orient="records", lines=True)
            item_descriptions_df = pd.read_json(DATASET_DIR + "/item_descriptions_short.jsonl", orient="records", lines=True)

            train_dataset = MakeTorchData(user_preferences_df, item_descriptions_df, train_df, train_df, is_training=True)
            eval_dataset = MakeTorchData(user_preferences_df, item_descriptions_df, train_df, valid_df, is_training=False)
            test_dataset = MakeTorchData(user_preferences_df, item_descriptions_df, train_df, test_df, is_training=False)
        

        total_reviews = len(train_dataset) + len(eval_dataset) + len(test_dataset)
        print("Total reviews:", total_reviews)
        
        print("Calculating token count statistics")
        max_context_window_length = 0
        # Total length: https://stackoverflow.com/questions/50431139/what-does-tqdms-total-parameter-do
        for prompt in tqdm(train_dataset, total=len(train_dataset)):
            len_prompt = len(prompt["input_ids"])
            max_context_window_length = max(max_context_window_length, len_prompt)

        '''
        args.context_window = 100
        for prompt in tqdm(train_dataset):
            print(prompt)
            assert False
        '''

        for prompt in tqdm(eval_dataset, total=len(eval_dataset)):
            len_prompt = len(prompt["input_ids"])
            max_context_window_length = max(max_context_window_length, len_prompt)

        for prompt in tqdm(test_dataset, total=len(test_dataset)):
            len_prompt = len(prompt["input_ids"])
            max_context_window_length = max(max_context_window_length, len_prompt)

        if RANDOM_SAMPLE:
            # To add margin of safety due to random dropping of reviews leading to context window length differences
            max_context_window_length += 4 

        print("Maximum number of tokens across all reviews: %d" % max_context_window_length)

        if max_context_window_length > 2048 and args.llm == "tinyllama":
            max_context_window_length = 2048
            print("TinyLlama context window limit exceeded, truncating to 2048 tokens")

        if args.context_window == -1:
            args.context_window = max_context_window_length

        '''
        train_class_weights = np.array([0, 0, 0, 0, 0])
        for label in tokenized_datasets["train"]["label"]:
            if REGRESSION:
                train_class_weights[int(label)-1] += 1
            else:
                train_class_weights[int(label)] += 1
        print(train_class_weights)
        train_class_weights = 1 / np.maximum(train_class_weights, 0.1)
        print(train_class_weights)
        '''
        
        use_full_shard = (args.is_distributed and (args.llm == "llama3:70b" or args.llm == "qwen2:72b"))
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        if args.is_distributed and not use_full_shard:
            # https://pytorch.org/docs/stable/generated/torch.cuda.device_count.html
            effective_batch_size *= torch.cuda.device_count()
        print("Effective batch size:", effective_batch_size)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, num_labels=1 if REGRESSION else 5, torch_dtype=torch.float32 if use_full_shard else torch.bfloat16,
            problem_type="regression" if REGRESSION else "single_label_classification",
            # attn_implementation='flash_attention_2',
            device_map=None if args.is_distributed else "balanced")
        
        # quantization_config=eight_bit_config
        # According to documentation, this is necessary to accommodate the new token (maybe not, let's see)
        # https://huggingface.co/docs/transformers/en/main_classes/tokenizer
        # model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        # Disable use_cache warning
        model.config.use_cache = False 

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "score"]
        if args.train_embedding:
            target_modules.append("embed_tokens")

        peft_config = LoraConfig(
            target_modules = target_modules,
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False,
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=False
        )

        # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
        print(model)
        print("Number of parameters in LLM:", sum(p.numel() for p in model.parameters()))
        if args.lora_rank != -1:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        model_params = list(model.named_parameters())
        model_params_key = [key for (key, value) in model_params]
        model_params_value = [value for (key, value) in model_params]
        embedding_key = "model.embed_tokens.weight"
        # print(model_params_key)

        # Solution taken from
        # https://github.com/huggingface/transformers/issues/657#issuecomment-760116075
        embedding_params = [value for (key, value) in model_params if "emb" in key]
        non_embedding_params = [value for (key, value) in model_params if "emb" not in key]

        if not args.train_embedding:
            for key, value in model_params:
                # print(key)
                if key == embedding_key:
                    assert args.lora_rank == -1
                    value.requires_grad_(False)
                    # print(value)
            optim = torch.optim.AdamW(lr=args.learning_rate, params=[{'params': model_params_value}])
        else:
            optim = torch.optim.AdamW(
                params=[{'params': non_embedding_params}, {'params': embedding_params, 'lr': args.learning_rate / 10}],
                lr=args.learning_rate,
            )

        is_small_dataset = total_reviews <= 80000
        training_args = TrainingArguments(
            output_dir="test_trainer_" + str(args.instance_id),
            save_strategy="epoch" if is_small_dataset else "steps",
            save_steps=1000,
            eval_strategy="epoch" if is_small_dataset else "steps",
            eval_steps=1000,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_strategy='steps',
            logging_steps=args.logging_steps,
            save_total_limit=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': not args.train_embedding},
            ddp_find_unused_parameters=True,
            local_rank=args.local_rank,
            lr_scheduler_type="constant",
            load_best_model_at_end=True,
            accelerator_config={
                "dispatch_batches": True,
            },
            fsdp="full_shard" if use_full_shard else "",
            greater_is_better=False,
            metric_for_best_model="eval_" + args.loss_function,
        )
        
        # train_class_weights = torch.tensor(train_class_weights, dtype=torch.bfloat16, requires_grad=False).to(device)

        # Subclass code modified from https://medium.com/deeplearningmadeeasy/how-to-use-a-custom-loss-with-hugging-face-fc9a1f91b39b
        class CustomTrainer(Trainer):
            '''
            # This is for classification
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=train_class_weights)
                return (loss, outputs) if return_outputs else loss
            '''

            def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
                labels = inputs.pop("labels")
                # labels = inputs.pop("label")
                outputs = model(**inputs)
                # print(labels, outputs)
                if REGRESSION:
                    pred = outputs.logits.ravel()
                    if args.loss_function == "mse":
                        loss = torch.nn.functional.mse_loss(pred.to(dtype=torch.float64), labels.to(dtype=torch.float64))
                    elif args.loss_function == "mae":
                        loss = torch.nn.functional.l1_loss(pred.to(dtype=torch.float64), labels.to(dtype=torch.float64))
                    else:
                        assert False
                    '''
                    loss = torch.nn.functional.smooth_l1_loss(
                        pred.to(dtype=torch.float64), labels.to(dtype=torch.float64), beta=LOSS_LINEAR_THRESHOLD)
                    '''
                    loss = loss.to(dtype=torch.float32)
                else:
                    logits = outputs.logits
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                return (loss, outputs) if return_outputs else loss
            
            # Due to a bug where HuggingFace data collator automatically transforms the regression target to float32,
            # a new dataloader creation method has to be created
            # Code based on https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L871
            # License: Apache 2.0 (only applies to the CustomTrainer code below this point)
            # The code has been modified to use the PyTorch dataloader instead.
            '''
            Copyright 2024 Hugging Face

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.
            '''
            def get_train_dataloader(self):
                if self.train_dataset is None:
                    raise ValueError("Trainer: training requires a train_dataset.")

                train_dataset = self.train_dataset

                dataloader_params = {
                    "batch_size": self._train_batch_size,
                    "collate_fn": torch.utils.data.default_collate,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "persistent_workers": self.args.dataloader_persistent_workers,
                    "sampler": self._get_train_sampler(),
                    "drop_last": self.args.dataloader_drop_last,
                    "prefetch_factor": self.args.dataloader_prefetch_factor
                }

                # print(self.train_dataset.labels)
                # print(next(iter(DataLoader(train_dataset))))
                return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


        def compute_metrics(eval_pred):
            if REGRESSION:
                pred, labels = eval_pred
                pred = np.reshape(pred, (np.shape(pred)[0]))
                pred = pred.astype(np.float64)
                labels = labels.astype(np.float64)
                # print(np.shape(pred), np.shape(labels))
                result = {}
                result["mse"] = np.mean((pred - labels)**2)
                result["rmse"] = np.sqrt(result["mse"])
                result["mae"] = np.mean(np.abs(pred - labels))
                return result
            else:
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)

                # This code courtesy of 
                # https://discuss.huggingface.co/t/transform-logits-to-probabilities-doesnt-work/14792/2
                probabilities = softmax(logits, axis=-1)
                predictions_mse = probabilities @ np.arange(5)

                result = {}
                # result["accuracy"] = metrics["accuracy"].compute(predictions=predictions, references=labels)["accuracy"]
                result["mse_most_likely_rating"] = np.mean((predictions - labels)**2)
                result["mse_expected_rating"] = np.mean((predictions_mse - labels)**2)
                result["macro_f1"] = f1_score(y_true=labels, y_pred=predictions, average="macro")
                # print(predictions[:10], predictions_mse[:10], labels[:10])
                return result

        # model = torch.compile(model, fullgraph=True)
        trainer = CustomTrainer(
            optimizers=(optim, None),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )

        # print("model.config.problem_type is %s" % model.config.problem_type)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        valid_metrics = trainer.evaluate()
        print("Final validation set metrics:", valid_metrics)
        eval_results = trainer.predict(test_dataset)
        print("Final test set metrics:", eval_results.metrics)
        if REGRESSION:
            predictions = eval_results.predictions
        else:
            predictions = np.argmax(eval_results.predictions, axis=-1)
            print(np.unique(predictions, return_counts=True))
        print("Original predictions:" + " " + str(predictions[:10].reshape(-1)))
        print("Ground truth labels:" + " " + str(eval_results.label_ids[:10]))

        # Evaluation of user and item training frequency MSE

        # Compute training frequency
        # Required for evaluation of MSE by item training frequency
        train_user_ids = train_dataset.test_df["userID"].tolist()
        train_item_ids = train_dataset.test_df["itemID"].tolist()
        test_user_ids = test_dataset.test_df["userID"].tolist()
        test_item_ids = test_dataset.test_df["itemID"].tolist()

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
        train_user_freq_sum = np.zeros(max_train_user_freq+1)
        train_item_freq_sum = np.zeros(max_train_item_freq+1)
        
        train_user_freq_list = np.zeros(len(test_dataset))
        train_item_freq_list = np.zeros(len(test_dataset))
        user_rating_count_list = np.zeros(len(test_dataset))
        item_rating_count_list = np.zeros(len(test_dataset))
        for i in range(len(test_dataset)):
            user_freq = train_user_freq[test_user_ids[i]]
            item_freq = train_item_freq[test_item_ids[i]]
            contribution = ((predictions[i] - eval_results.label_ids[i]) ** 2)[0]
            train_user_freq_sum[user_freq] += contribution
            train_item_freq_sum[item_freq] += contribution
            train_user_freq_count[user_freq] += 1
            train_item_freq_count[item_freq] += 1
            train_user_freq_list[i] = user_freq
            train_item_freq_list[i] = item_freq

            result = test_dataset.get_attr(i)
            # print(result)
            # assert False
            user_rating_count_list[i] = result["user_rating_count"]
            item_rating_count_list[i] = result["item_rating_count"]

        results_df = pd.DataFrame({
            "train_user_freq": train_user_freq_list, 
            "train_item_freq": train_item_freq_list, 
            "user_rating_count": user_rating_count_list,
            "item_rating_count": item_rating_count_list,
            "predicted": predictions.flatten(),
            "actual": eval_results.label_ids.flatten()})
        results_df.to_json("%s.jsonl" % str(args.outfile), orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default=LLM, dest='llm', type=str)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, dest='dataset_dir', type=str)
    parser.add_argument('--instance_id', default=INSTANCE_ID, dest='instance_id', type=str)
    parser.add_argument('--outfile', default=OUTFILE, dest='outfile', type=str)
    parser.add_argument('--context_window', default=CONTEXT_WINDOW, dest='context_window', type=int)
    parser.add_argument('--batch_size', default=BATCH_SIZE, dest='batch_size', type=int)
    parser.add_argument('--lora_rank', default=LORA_RANK, dest='lora_rank', type=int)
    parser.add_argument('--lora_alpha', default=LORA_ALPHA, dest='lora_alpha', type=float)
    parser.add_argument('--lora_dropout', default=LORA_DROPOUT, dest='lora_dropout', type=float)
    parser.add_argument('--learning_rate', default=LEARNING_RATE, dest='learning_rate', type=float)
    parser.add_argument('--gradient_accumulation_steps', default=GRADIENT_ACCUMULATION_STEPS, 
                        dest='gradient_accumulation_steps', type=int)
    parser.add_argument('--gradient_checkpointing', default=GRADIENT_CHECKPOINTING, 
                        dest='gradient_checkpointing', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--logging_steps', default=LOGGING_STEPS, dest='logging_steps', type=int)
    parser.add_argument('--train_embedding', default=TRAIN_EMBEDDING, 
                        dest='train_embedding', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_epochs', default=NUM_EPOCHS, dest='num_epochs', type=int)
    parser.add_argument('--resume_from_checkpoint', default=RESUME_FROM_CHECKPOINT, dest='resume_from_checkpoint', 
                        type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--show_token_stats', default=SHOW_TOKEN_STATS, dest='show_token_stats', 
                        type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stopping_patience', default=EARLY_STOPPING_PATIENCE, 
                        dest='early_stopping_patience', type=int)
    
    # From preprocess.py
    parser.add_argument('--max_user_reviews', default=MAX_USER_REVIEWS, dest='max_user_reviews', type=int)
    parser.add_argument('--max_item_reviews', default=MAX_ITEM_REVIEWS, dest='max_item_reviews', type=int)

    parser.add_argument('--summarized', default=SUMMARIZED, dest='summarized', 
                        type=bool, action=argparse.BooleanOptionalAction)
    
    # How to enforce choice values
    # https://stackoverflow.com/questions/25295487/python-argparse-value-range-help-message-appearance
    parser.add_argument('--add_ratings', default=ADD_RATINGS, dest='add_ratings', choices=range(3), type=int)
    
    parser.add_argument('--hide_reviews', default=HIDE_REVIEWS, dest='hide_reviews',
                        type=bool, action=argparse.BooleanOptionalAction)
    
    parser.add_argument('--random_sample', default=RANDOM_SAMPLE, dest='random_sample', 
                        type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--loss_function', default=LOSS_FUNCTION, dest='loss_function', type=str)
    args = parser.parse_args()
    print(args)

    LLM = args.llm
    DATASET_DIR = args.dataset_dir
    INSTANCE_ID = args.instance_id
    OUTFILE = args.outfile
    CONTEXT_WINDOW = args.context_window
    BATCH_SIZE = args.batch_size
    LORA_RANK = args.lora_rank
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    LEARNING_RATE = args.learning_rate
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    LOGGING_STEPS = args.logging_steps
    TRAIN_EMBEDDINGS = args.train_embedding
    NUM_EPOCHS = args.num_epochs
    RESUME_FROM_CHECKPOINT = args.resume_from_checkpoint
    SHOW_TOKEN_STATS = args.show_token_stats
    SUMMARIZED = args.summarized
    LOSS_FUNCTION = args.loss_function
    RANDOM_SAMPLE = args.random_sample
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    # From preprocess.py
    MAX_USER_REVIEWS = args.max_user_reviews
    MAX_ITEM_REVIEWS = args.max_item_reviews
    ADD_RATINGS = args.add_ratings
    HIDE_REVIEWS = args.hide_reviews

    main(args)
