import json
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import numpy as np
from itertools import product
import random
import concurrent.futures
import time
import os
import word_variable

# MAKING PROMPTS

## Kuperman's list
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()
kuperman_word_list = df_test.Word.tolist()
len(kuperman_word_list)

## extract random words (n=50)
def extract_random_words_from_list(word_list, n=10000, random_seed=0):
    np.random.seed(random_seed)
    nums = np.random.choice(range(0, len(kuperman_word_list)), n, replace=False)
    result = [word_list[k] for k in nums]
    return result

def split_into_groups(word_list, ratios=[0.8, 0.2]):
    """
    비복원추출을 통해 주어진 범위의 정수를 주어진 비율에 따라 n개의 그룹으로 나눕니다.

    :param range_start: 추출의 시작 범위
    :param range_end: 추출의 끝 범위
    :param ratios: 세 그룹의 비율 (예: [0.4, 0.3, 0.3]는 40%, 30%, 30%로 나눕니다)
    :return: 세 그룹의 정수 리스트
    """

    total_numbers = len(word_list)
    group_sizes = [int(total_numbers * ratio) for ratio in ratios]
    print("each_group_n: ", group_sizes)

    # 만약 비율이 완벽히 맞지 않아서 모든 수가 할당되지 않았다면, 마지막 그룹에 할당합니다.
    diff = total_numbers - sum(group_sizes)
    group_sizes[-1] += diff

    all_numbers = list(range(total_numbers))
    random.shuffle(all_numbers)

    k=0
    each_group_number = list()
    for group_size in group_sizes:
        each_group_number.append(all_numbers[k: (k+group_size)])
        k += group_size

    grouped_word_list = list()
    for a_group_numbers in each_group_number:
        grouped_word_list.append([word_list[k] for k in a_group_numbers])

    return tuple(grouped_word_list)


# iterable = list(product(*[word_list, word_list]))
def generate_eng_word_prompt(word_pair):
    word1 = word_pair[0]
    word2 = word_pair[1]
    # result = list()
    # print(word1, word2)
    json_data = dict()
    aoa_word1 = df_test['Rating.Mean'][df_test.Word == word1].tolist()[0]
    aoa_word2 = df_test['Rating.Mean'][df_test.Word == word2].tolist()[0]
    # print("ongoing")
    # print(aoa_word1, aoa_word2)
    if aoa_word1 < aoa_word2:
        assist_output = word1
    else:
        assist_output = word2
    # print(word1, ":", aoa_word1, "&", word2, ":", aoa_word2, "->", assist_output)
    json_data['text'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Which one do you learn earlier? {word1} or {word2}?. ### Response: {assist_output}"

    return json_data

extracted_wlist = extract_random_words_from_list(kuperman_word_list, 125, random_seed=0)
train, test = split_into_groups(extracted_wlist, ratios=[0.8, 0.2])

def make_pair(word_list):
    res = list()
    for x in tqdm(product(*[word_list, word_list])):
        res.append(x)
    return res

train_pair = make_pair(train)
test_pair = make_pair(test)

train_pair = [x for x in train_pair if x[0]!=x[1]]
test_pair = [x for x in test_pair if x[0]!=x[1]]


max_workers = os.cpu_count()
max_workers

def run(pair_list, update_frequency=1000):
    with tqdm(total=len(pair_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results=[]
            for k in range(len(pair_list)):
                results.append(executor.submit(generate_eng_word_prompt, pair_list[k]))

                if k % update_frequency == 0:
                    pbar.update(update_frequency)

            # results = list(tqdm(executor.map(generate_eng_word_prompt,
            #                                  pair_list), total=len(pair_list), miniters=1000))
            return results

train_json = run(train_pair)
test_json = run(test_pair)

train_res=[x.result() for x in tqdm(train_json)]
test_res=[x.result() for x in tqdm(test_json)]

train_list = [x['text'] for x in tqdm(train_res)]
test_list = [x['text'] for x in tqdm(test_res)]

pd.DataFrame(test_list, columns=['text']).to_csv('test_25.csv', index=False)
pd.DataFrame(train_list, columns=['text']).to_csv('train_100.csv', index=False)







# -------------------Below to be deleted-------------------



# with open("llama_train_data_3000words.json", "w", encoding="utf-8") as f:
#     json.dump(train_res, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장
#
# with open("llama_test_data_3000words.json", "w", encoding="utf-8") as f:
#     json.dump(test_res, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장





## PRACTICE LOAD DATASET

from datasets import load_dataset

#dataset_name = "timdettmers/openassistant-guanaco" ###Human ,.,,,,,, ###Assistant

issues_dataset = load_dataset("json", data_files="llama_train_data_3000words.json", split="train")
issues_dataset[0]


dataset_name = 'AlexanderDoria/novel17_test' #french novels
dataset = load_dataset(dataset_name, data_files="novel17_eval.jsonl", split="test")
dataset[0]
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses,
})

# Split the data into train and test sets, with 90% in the train set
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save the dataframes to .jsonl files
train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.json', orient='records', lines=True)