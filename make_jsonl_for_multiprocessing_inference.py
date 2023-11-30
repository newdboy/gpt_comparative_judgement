# import jsonlines
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from itertools import combinations

# extract random words (n=50)


#12high&low
word_list_kims_12high = extract_random_words_from_list(df_12high, 20, random_seed=0)
word_list_kims_12low = extract_random_words_from_list(df_12low, 20, random_seed=0)

#34high&low
word_list_kims_34high = extract_random_words_from_list(df_34high, 20, random_seed=0)
word_list_kims_34low = extract_random_words_from_list(df_34low, 20, random_seed=0)

#56high&low
word_list_kims_56high = extract_random_words_from_list(df_56high, 20, random_seed=0)
word_list_kims_56low = extract_random_words_from_list(df_56low, 20, random_seed=0)

# import past compared list
# df_have_compared = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/090723_cj_200words_kuperman.xlsx')
# df_have_compared.head()
# wlist1 = df_have_compared['word1'].tolist()
# wlist2 = df_have_compared['word2'].tolist()


def make_list_for_inference(word_list, model="gpt-3.5-turbo", inference_type="comb"):

    def get_two_element_combinations(lst):
        return list(combinations(lst, 2))

    if inference_type == "comb":
        # combination of two words
        comparing_list = get_two_element_combinations(word_list)
    elif inference_type == "perm":
        # permutation of two words
        comparing_list = list(product(*[word_list, word_list]))

    result = list()
    for x in tqdm(comparing_list):
        word1, word2 = x[0], x[1]
        print(word1, word2)
        data_line = OrderedDict()
        data_line["model"] = model
        data_line["messages"] = [{"role": "system",
                                  "content": "Pick the one word you believe is learned first in childhood"
                                  },
                                 {"role": "user",
                                  "content": str(word1) + ", " + str(word2)
                                  }]
        data_line["temperature"] = 0
        data_line["max_tokens"] = 100
        result.append(data_line)
    return result

def make_list_for_inference_ko(word_list1, word_list2, model="gpt-3.5-turbo"):
    result = list()
    for word1, word2 in tqdm(list(product(*[word_list1, word_list2]))):
        data_line = OrderedDict()
        data_line["model"] = model
        data_line["messages"] = [{"role": "system",
                                  "content": "Pick the one word you believe is learned first in childhood"
                                  },
                                 {"role": "user",
                                  "content": str(word1) + ", " + str(word2)
                                  }]
        data_line["temperature"] = 0
        data_line["max_tokens"] = 100
        result.append(data_line)

    return result

def save_list_to_jsonl(comp_list, save_file_name="inference_data.jsonl"):
    with open(save_file_name, "w", encoding="utf-8") as f:
        for data_line in comp_list:
            json.dump(data_line, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장
            f.write("\n")

# model=ft:gpt-3.5-turbo-0613:pickler:flight-09101223:7x6HbsGs
mp_list = make_list_for_inference(word_list_kuperman_for_test, model="ft:gpt-3.5-turbo-0613:pickler:flight-1:7x0w43U1")

mp_list = make_list_for_inference(word_list_kuperman_for_test, model="ft:babbage-002:pickler:bab-en-100:82agtzsH")


mp_list12 = make_list_for_inference_ko(word_list_kims_12high, word_list_kims_12low, model="ft:gpt-3.5-turbo-0613:pickler:flight-1:7x0w43U1")  #12high&low
mp_list34 = make_list_for_inference_ko(word_list_kims_34high, word_list_kims_34low, model="ft:gpt-3.5-turbo-0613:pickler:flight-1:7x0w43U1")  #34high&low
mp_list56 = make_list_for_inference_ko(word_list_kims_56high, word_list_kims_56low, model="ft:gpt-3.5-turbo-0613:pickler:flight-1:7x0w43U1")  #56high&low

save_list_to_jsonl(mp_list12+mp_list34+mp_list56, save_file_name="inference_data_0911_ko_1.jsonl")

# # To get the tokenizer corresponding to a specific model in the OpenAI API:
# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4")
# enc

# SSL-certificate
# https://stackoverflow.com/questions/42098126/mac-osx-python-ssl-sslerror-ssl-certificate-verify-failed-certificate-verify

# EXECUTE BELOW IN TERMINAL
# python parallel_jury.py \
#   --requests_filepath target_data.jsonl \
#   --save_filepath result_data.jsonl \
#   --api_key [API-key] \
#   --request_url https://api.openai.com/v1/chat/completions \
#   --max_requests_per_minute 3200 \
#   --max_tokens_per_minute 85000 \
#   --token_encoding_name cl100k_base \
#   --max_attempts 5 \
#   --logging_level 20