import gpt_jury
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import concurrent.futures
from itertools import product

# retry in Kuperman
df_test = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_test.head()

# extract random words (n=50)
word_list_kuperman = [df_test.Word[k] for k in np.random.choice(range(0, len(df_test)+1), 10, replace = False)]
len(word_list_kuperman)


# normal method
dff_res=list()
for word1, word2 in tqdm(list(product(*[word_list_kuperman, word_list_kuperman]))):
    if word1 != word2:
        answer = gpt_jury.word_dff_jury(word1, word2, "gpt-4")  # not "gpt-4" "gpt-3.5-turbo
        dff_res.append([word1,
                        word2,
                        answer])


# NEW METHOD (multiprocessing)

def run(list_sum):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(gpt_jury.word_dff_jury,
                                         list_sum[0], list_sum[1]), total=len(list_sum[0])))
    return results


comp_list = [x for x in list(product(*[word_list_kuperman, word_list_kuperman])) if x[0]!=x[1]]
list_a = [k[0] for k in comp_list]
list_b = [k[1] for k in comp_list]
list_sum = [list_a, list_b]
len(list_a)

# run
test = run(list_sum)
test

# check reliability
test_2 = [k[2] for k in dff_res]
for x,y,z in zip(test, test_2, dff_res):
    if x!=y:
        print(x,y,z)

