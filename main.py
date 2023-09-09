import pandas as pd
import gpt_jury
import numpy as np
# import ftxt_embedding
import random
from tqdm import tqdm
from itertools import product

# import word list(Kim's)
test_source = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/김한샘어휘_20220902.xlsx'
df = pd.read_excel(test_source)
df.head()
df_dd = df.drop_duplicates(subset=['단어'])
len(df)
len(df_dd)
# compare
word_list_dd = df_dd['단어'].tolist()

test=['단어','어휘','문장','낱말','문법']
# (funny) better accuracy in chatgpt than Embedding Vector
# You are a machine that can compute cosine similarity.
# test=['단어','어휘','문장','낱말','문법']
# 1. Pick the two words with the highest similarity to the elements in the test.
# 2. Don't explain, just give me the correct answer.

df_12high = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '상')].tolist()
df_12middle = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '중')].tolist()
df_12low = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '하')].tolist()

# USING gpt-3-turbo
dff_res = list()
cost_sum3 = float()
for x in tqdm(range(10)):
    m = random.randint(0, len(df_12high))
    n = random.randint(0, len(df_12low))
    tmp, cost = gpt_jury.word_dff_jury(df_12high[m], df_12low[n])
    dff_res.append([df_12high[m], df_12low[n], tmp])  #, "gpt-4"
    cost_sum3 += cost

for k in dff_res:
    print(k)

# USING gpt-4
res_tmp = list()
cost_sum = float()
for k in dff_res:
    answer, cost = gpt_jury.word_dff_jury(k[0], k[1], "gpt-4")
    res_tmp.append([k[0], k[1], answer[0]])
    cost_sum += cost

cost_sum

for k in res_tmp:
    print(k)

#------------------------------------------------------------------------------
# retry in Kuperman
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()

# extract random words (n=50)
word_list_kuperman = [df_test.Word[k] for k in np.random.choice(range(0, len(df_test)+1), 200, replace = False)]
len(word_list_kuperman)

# gpt-based comparison
dff_res=list()
cost_sum=float()
for word1, word2 in tqdm(list(product(*[word_list_kuperman, word_list_kuperman]))):
    if word1 != word2:
        answer, cost = gpt_jury.word_dff_jury(word1, word2)  # not "gpt-4" "gpt-3.5-turbo
        dff_res.append([word1,
                        word2,
                        answer])
        cost_sum += cost
        print(cost_sum)



def check_true_ratio(dff_res):
    for k in range(len(dff_res)):
        if len(dff_res[k]) == 3:
            if dff_res[k][1] == dff_res[k][2]:
                dff_res[k].append(1)
            else:
                dff_res[k].append(0)

    sum_true = sum([k[3] for k in dff_res])
    return (sum_true)/(len(dff_res))


