# import jsonlines
import json
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import numpy as np
from itertools import product

# Kuperman's list
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()

# extract random words (n=50)
word_list_kuperman = [df_test.Word[k] for k in np.random.choice(range(0, len(df_test)+1), 10, replace = False)]
len(word_list_kuperman)


# import past compared list
# df_have_compared = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/090723_cj_200words_kuperman.xlsx')
# df_have_compared.head()
# wlist1 = df_have_compared['word1'].tolist()
# wlist2 = df_have_compared['word2'].tolist()

# make jsonl with data
for word1, word2 in tqdm(list(product(*[word_list_kuperman, word_list_kuperman]))):
    if word1 != word2:
        aoa_word1 = df_test['Rating.Mean'][df_test.Word == word1].tolist()[0]
        aoa_word2 = df_test['Rating.Mean'][df_test.Word == word2].tolist()[0]
        print(aoa_word1, aoa_word2)
        if aoa_word1 < aoa_word2:
            assist_output = word1
        else:
            assist_output = word2
        print(word1, ":", aoa_word1, "&", word2, ":", aoa_word2, "->", assist_output)
        data_line = OrderedDict()
        data_line["model"] = "gpt-3.5-turbo"
        data_line["messages"] = [{"role": "system",
                                  "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                  },
                                 {"role": "user",
                                  "content": str(word1) + ", " + str(word2)
                                  },
                                 {"role": "assistant",
                                  "content": str(assist_output)
                                  },
                                 ]
        data_line["epoch"] = 1
        with open("training_data.jsonl", "a", encoding="utf-8") as f:
            json.dump(data_line, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장
            f.write("\n")
