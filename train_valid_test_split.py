import pandas as pd
import random
from sklearn.model_selection import train_test_split
import argparse
import os
import html

MODEL = "Meta-Llama-3-8B-Instruct-fp16" # model path, 不要加上 .gguf
JSON_INPUT_FILE = "Musical_Instruments_5.json" # dataset file path
SAVE_DIRECTORY = "musical_instruments_5_2014_test" # 保存 preprocessed 数据集的地方

MAX_USER_REVIEWS = 15
MAX_ITEM_REVIEWS = 15
LOG_FREQ = 1000
RANDOM_SEED = 297370238

# Parameters that cannot be set from command line yet
TRAIN_RATE = 0.8
RANDOM_SEED_SELECT = 505303486
RANDOM_SEED_TRAIN = 984444699
RANDOM_SEED_TEST = 636893143

def preprocess(df):
    user_row_ids = [[] for i in range(n_users)]
    item_row_ids = [[] for i in range(n_items)] 
    userID_column = df["userID"]
    itemID_column = df["itemID"]
    n_rows = df.shape[0]
    df = df.assign(reviewID = [i for i in range(n_rows)])
    user_df = pd.DataFrame({"userID": [i for i in range(n_users)]})
    user_df["reviewIDs"] = ""
    item_df = pd.DataFrame({"itemID": [i for i in range(n_items)]})
    item_df["reviewIDs"] = ""
    
    for i in range(n_users):
        assert user_df.at[user_df.index[i], "userID"] == i
        assert user_df.iloc[i]["userID"] == i

    for i in range(n_items):
        assert item_df.at[item_df.index[i], "itemID"] == i
        assert item_df.iloc[i]["itemID"] == i

    for i in range(n_rows):
        assert df.at[df.index[i], "reviewID"] == i
        assert df.iloc[i]["reviewID"] == i

    for i in range(n_rows):
        userID = userID_column.iloc[i]
        itemID = itemID_column.iloc[i]
        user_row_ids[userID].append(i)
        item_row_ids[itemID].append(i)
        df.at[df.index[i], "reviewID"] = i

    for i in range(n_users):
        userID = userID_column.iloc[i]
        user_review_list = user_row_ids[i].copy()
        user_review_list = random.sample(user_review_list, min(MAX_USER_REVIEWS, len(user_review_list)))
        user_df.at[user_df.index[i], "reviewIDs"] = ",".join(list(map(str, user_review_list)))

        if (i+1) % LOG_FREQ == 0:
            print("User %d of %d: %d user review(s) in training set" % (i+1, n_users, len(user_review_list)))

    for i in range(n_items):
        itemID = itemID_column.iloc[i]
        item_review_list = item_row_ids[i].copy()
        item_review_list = random.sample(item_review_list, min(MAX_ITEM_REVIEWS, len(item_review_list)))
        item_df.at[item_df.index[i], "reviewIDs"] = ",".join(list(map(str, item_review_list)))
        
        if (i+1) % LOG_FREQ == 0:
            print("Item %d of %d: %d item review(s) in training set" % (i+1, n_items, len(item_review_list)))
    return df, user_df, item_df

def unescape_reviews(df):
    df["reviewText"] = df["reviewText"].apply(html.unescape)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_user_reviews', default=MAX_USER_REVIEWS, dest='max_user_reviews', type=int)
    parser.add_argument('--max_item_reviews', default=MAX_ITEM_REVIEWS, dest='max_item_reviews', type=int)
    parser.add_argument('--log_freq', default=LOG_FREQ, dest='log_freq', type=int)
    parser.add_argument('--json_input_file', default=JSON_INPUT_FILE, dest='json_input_file', type=str)
    parser.add_argument('--save_directory', default=SAVE_DIRECTORY, dest='save_directory', type=str)
    args = parser.parse_args()
    print(args)

    MAX_USER_REVIEWS = args.max_user_reviews
    MAX_ITEM_REVIEWS = args.max_item_reviews
    LOG_FREQ = args.log_freq
    JSON_INPUT_FILE = args.json_input_file
    SAVE_DIRECTORY = args.save_directory

    JSON_INPUT_FILE = "data/" + JSON_INPUT_FILE
    SAVE_DIRECTORY = "data/" + SAVE_DIRECTORY

    df = pd.read_json(JSON_INPUT_FILE, lines=True)
    df.drop(columns=['reviewerName', 'reviewTime'], inplace=True)

    for column in ["unixReviewTime", "helpful", "summary"]:
        df.drop(column, inplace=True, axis='columns')
    
    # df.drop_duplicates(inplace=True)
    df = df[df['reviewText'] != '']

    df['reviewerID'] = df['reviewerID'].astype("category").cat.codes
    df['asin'] = df['asin'].astype("category").cat.codes # product id
    df.rename(columns={'asin': 'itemID', 'reviewerID': 'userID', 'overall': 'rating'}, inplace=True)
    n_users = len(set(df['userID']))
    n_items = len(set(df['itemID']))
    n_rows = df.shape[0]
    print("Users: %d; Items: %d; Reviews: %d (after removing empty review text)" % (n_users, n_items, n_rows))
    random.seed(RANDOM_SEED_SELECT) # For reproducibility in generating training, validation and test set

    train, valid = train_test_split(df, test_size=1 - TRAIN_RATE, random_state=RANDOM_SEED_TRAIN)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=RANDOM_SEED_TEST)

    train, user_df, item_df = preprocess(train)
    train = unescape_reviews(train)
    valid = unescape_reviews(valid)
    test = unescape_reviews(test)

    # https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
    # Create directory if not exists
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    # Remove users and items that do not appear in the training set
    empty_user_df = user_df[user_df["reviewIDs"] == ""]
    empty_item_df = item_df[item_df["reviewIDs"] == ""]

    empty_users = []
    for i in range(len(empty_user_df)):
        empty_users.append(empty_user_df.iloc[i]["userID"])

    empty_items = []
    for i in range(len(empty_item_df)):
        empty_items.append(empty_item_df.iloc[i]["itemID"])

    print("Number of users without reviews in training set: %d," % len(empty_users), empty_users)
    print("Number of items without reviews in training set: %d," % len(empty_items), empty_items)

    valid = valid[~valid["userID"].isin(empty_users)]
    valid = valid[~valid["itemID"].isin(empty_items)]
    test = test[~test["userID"].isin(empty_users)]
    test = test[~test["itemID"].isin(empty_items)]

    train.to_json(SAVE_DIRECTORY + "/train_raw.jsonl", orient="records", lines=True)
    valid.to_json(SAVE_DIRECTORY + "/valid_raw.jsonl", orient="records", lines=True)
    test.to_json(SAVE_DIRECTORY + "/test_raw.jsonl", orient="records", lines=True)
    user_df.to_json(SAVE_DIRECTORY + "/user.jsonl", orient="records", lines=True)
    item_df.to_json(SAVE_DIRECTORY + "/item.jsonl", orient="records", lines=True)