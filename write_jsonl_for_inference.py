# import jsonlines
import json
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

# Kuperman's list
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()

# extract random words (n=50)
word_list_kuperman = [df_test.Word[k] for k in np.random.choice(range(0, len(df_test)+1), 50, replace = False)]
len(word_list_kuperman)

# import past compared list
# df_have_compared = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/090723_cj_200words_kuperman.xlsx')
# df_have_compared.head()
# wlist1 = df_have_compared['word1'].tolist()
# wlist2 = df_have_compared['word2'].tolist()

# make jsonl with data
for word1, word2, in zip(wlist1, wlist2):
    data_line = OrderedDict()
    data_line["model"] = "gpt-3.5-turbo"
    data_line["messages"] = [{"role": "system",
                              "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                              },
                             {"role": "user",
                              "content": str(word1) + ", " + str(word2)
                              }]
    data_line["temperature"] = 0
    data_line["max_tokens"] = 100
    with open("target_data.jsonl", "a", encoding="utf-8") as f:
        json.dump(data_line, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장
        f.write("\n")


# # To get the tokeniser corresponding to a specific model in the OpenAI API:
# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4")
# enc

# SSL-certificate
# https://stackoverflow.com/questions/42098126/mac-osx-python-ssl-sslerror-ssl-certificate-verify-failed-certificate-verify

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