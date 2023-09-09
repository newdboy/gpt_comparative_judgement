import multiprocessing
import time
import concurrent.futures
import pandas as pd
import gpt_jury
import numpy as np
from tqdm import tqdm
from itertools import product

# import in Kuperman
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()

# extract random words (n=50)
word_list_kuperman = [df_test.Word[k] for k in np.random.choice(range(0, len(df_test)+1), 50, replace = False)]
len(word_list_kuperman)

# multiprocessing.cpu_count()
# n_cpu = int(multiprocessing.cpu_count() * 0.7) # 내 컴퓨터 CPU 코어 수 * 사용량(0.7)
# n_cpu

def run(list_sum):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(gpt_jury.word_dff_jury,
                                         list_sum[0], list_sum[1]), total=len(list_sum[0])))
    return results

# NEW METHOD (multiprocessing)
comp_list = [x for x in list(product(*[word_list_kuperman, word_list_kuperman])) if x[0]!=x[1]]
list_a = [k[0] for k in comp_list]
list_b = [k[1] for k in comp_list]
list_sum = [list_a, list_b]
len(list_a)

# run
test = run(list_sum)

pd.DataFrame(test).to_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/test.xlsx')
test[1].split('.')
import re
test = dff_res[:]
tmp = [x[2] for x in test]
test[0][0] in tmp[0]
res = list()
for k in range(len(test)):
    if test[k][0] in tmp[k]:
        res.append(0)
    else:
        res.append(1)
res2 = list()
for word1, word2 in zip([x[0] for x in test], [x[1] for x in test]):
    wordA_AoA = df_test['Rating.Mean'][df_test.Word == word1].tolist()[0]
    wordB_AoA = df_test['Rating.Mean'][df_test.Word == word2].tolist()[0]
    if wordA_AoA > wordB_AoA:
        res2.append(1)
    else:
        res2.append(0)
sum([1 for x,y in zip(res, res2) if x==y])/len(res)

# post-processing
dff_res_=list()
for k in range(len(test)):
    # print(list_a[k], list_b[k], test[k])
    _ = re.findall(r'(\w+|\d+%)', test[k])
    tmp = [list_a[k], list_b[k]] + _
    if tmp[1] == tmp[2]:
        tmp.append(1)
    else:
        tmp.append(0)
    dff_res_.append(tmp)

# add boolean if posterior is easier
df_save1 = pd.DataFrame(dff_res_,
                        columns=['word1', 'word2', 'easier_word', 'confidence', 'if_word2_is_easier'])
df_save1.to_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/cj_50words_kuperman_090801.xlsx')
len(df_save1)
df_test[df_test.Word.isin(word_list_kuperman)].to_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/50words_kuperman_090801.xlsx')


# normal method
dff_res=list()
for word1, word2 in tqdm(list(product(*[word_list_kuperman, word_list_kuperman]))):
    if word1 != word2:
        answer = gpt_jury.word_dff_jury(word1, word2)  # not "gpt-4" "gpt-3.5-turbo
        dff_res.append([word1,
                        word2,
                        answer])

test_2 = [k[2] for k in dff_res]
for x,y,z in zip(test, test_2, dff_res):
    if x!=y:
        print(x,y,z)