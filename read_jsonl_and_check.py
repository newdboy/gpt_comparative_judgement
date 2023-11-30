import jsonlines
from pprint import pprint
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm

# Convert GPT's Comparative Judgement data(from API's) to XLSX
def convert_cj_result_to_xlsx(result_data_dir, xlsx_save_dir):
    with jsonlines.open(result_data_dir) as f:
        lines = [x for x in f.iter()]

    parallel_jurisdiction_list=[]
    for k in range(len(lines)):
        check_if_noerror = type(lines[k][1]) == type(dict())
        if check_if_noerror == True:
            input = lines[k][0]['messages'][1]['content']
            input_words_list = [x.strip() for x in input.split(',')]
            result = lines[k][1]['choices'][0]['message']['content']
            parallel_jurisdiction_list.append(input_words_list+[result])

    pd.DataFrame(parallel_jurisdiction_list, columns=['word1', 'word2', 'easier_word']).to_excel(xlsx_save_dir)

result_data_dir = "/Users/kintch/PycharmProjects/gpt_comparativ_judgement/result_data.jsonl"
xlsx_save_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/test_3.xlsx'
convert_cj_result_to_xlsx(result_data_dir, xlsx_save_dir)

convert_cj_result_to_xlsx("/Users/kintch/PycharmProjects/gpt_comparativ_judgement/inference_data_result_0911_ko_1.jsonl",
                          "/Users/kintch/PycharmProjects/gpt_comparativ_judgement/inference_data_result_0911_ko_1.xlsx")


# Check before FINE-TUNE
data_path = "train_set.jsonl"

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]
len(dataset)

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)

# Format error checks
def check_data_format_validation(dataset):
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

check_data_format_validation(dataset)



# Check after FINE-TUNE
df=pd.read_csv('/Users/kintch/file.csv')
df

df=pd.read_excel('/Users/kintch/PycharmProjects/gpt_comparativ_judgement/inference_data_result_091013_4_finetune_model-2.xlsx')
df.head()

len(df)

# Check inconsistency
inconsistent_cj_words = list()
consistent_cjs = list()
for k in tqdm(range(len(df))):
    w1 = df.iloc[k].word1
    w2 = df.iloc[k].word2
    easier = df.iloc[k].easier_word
    # print(w1, w2)
    try:
        inverse_case_easier = df.easier_word[(df['word2'] == w1) & (df['word1'] == w2)]
        inverse_case_easier_word = inverse_case_easier.tolist()[0]
        if inverse_case_easier_word != easier:
            inconsistent_cj_words.append((w1, w2))
        else:
            consistent_cjs.append([w1, w2, easier])
    except:
        print("ERROR!", w2, w2, "LOST!")

pd.DataFrame(consistent_cjs, columns=['word1', 'word2', 'easier_word']).to_excel('/Users/kintch/PycharmProjects/gpt_comparativ_judgement/inference_data_result_091013_4_finetune_model-2_incon_deleted.xlsx')


# Check inconsistent word's AoA difference

word_list_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx'
word_list = pd.read_excel(word_list_dir)

def absolute_value(num):
    if num < 0:
        return -num
    else:
        return num

inconsist_words_aoa_diff=list()
for k in inconsistent_cj_words:
    word1, word2 = k[0], k[1]

    word1_AoA = word_list['Rating.Mean'][word_list.Word == word1].tolist()[0]
    word2_AoA = word_list['Rating.Mean'][word_list.Word == word2].tolist()[0]

    diff = absolute_value(word1_AoA-word2_AoA)
    inconsist_words_aoa_diff.append(diff)

import pickle
with open('data09111425.pickle', 'wb') as file:
    pickle.dump(inconsist_words_aoa_diff, file)

# 오류 -> Code interpreter
# import matplotlib.pyplot as plt
#
# plt.hist(inconsist_words_aoa_diff, bins='auto')
#
# # Setting the title and labels
# plt.title('Number Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
#
# # Displaying the plot
# plt.show()