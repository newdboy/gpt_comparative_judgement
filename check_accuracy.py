import pandas as pd
from tqdm import tqdm

# compare between AoA metrix & comparative judgement

word_list_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx'
word_list = pd.read_excel(word_list_dir)

compared_words_list_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/test_3.xlsx'
compared_words_list = pd.read_excel(compared_words_list_dir)

compared_words_list.head()

def calc_accuracy(compared_words_list, word_list):
    if_word2_is_easier = list()
    for k in tqdm(range(len(compared_words_list))):
        if compared_words_list['word2'].iloc[k] == compared_words_list['easier_word'].iloc[k]:
            if_word2_is_easier.append(1)
        else:
            if_word2_is_easier.append(0)

    compared_words_list['if_word2_is_easier'] = if_word2_is_easier

    result=[]
    for k in tqdm(range(len(compared_words_list))):
        wordA = compared_words_list['word1'].iloc[k]
        wordB = compared_words_list['word2'].iloc[k]

        wordA_AoA = word_list['Rating.Mean'][word_list.Word == wordA].tolist()[0]
        wordB_AoA = word_list['Rating.Mean'][word_list.Word == wordB].tolist()[0]

        if wordA_AoA > wordB_AoA:
            result.append(1)
        else:
            result.append(0)

    tmp1 = sum([1 for x,y in zip(compared_words_list['if_word2_is_easier'].tolist(), result) if x==y])
    tmp2 = len(result)
    return (tmp1/tmp2)


# compare between comp.judgement(1st) vs comp.judgement(2nd) vs ...

compared_words_list_1_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/test_3.xlsx'
compared_words_list_1 = pd.read_excel(compared_words_list_1_dir)

compared_words_list_2_dir = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/090723_cj_200words_kuperman.xlsx'
compared_words_list_2 = pd.read_excel(compared_words_list_2_dir)

for k in range(len(compared_words_list_2)):
    if compared_words_list_2.easier_word.iloc[k] == 'Word A':
        compared_words_list_2.easier_word.iloc[k] = compared_words_list_2.word1.iloc[k][:]
    elif compared_words_list_2.easier_word.iloc[k] == 'Word B':
        compared_words_list_2.easier_word.iloc[k] = compared_words_list_2.word2.iloc[k][:]

# check if disordered
for k in range(len(compared_words_list_1)):
    if compared_words_list_1.word1.iloc[k] == compared_words_list_2.word1.iloc[k]:
        pass
    else:
        print("ERROR!")

# order the words
compared_words_list_1.sort_values(by=['word1', 'word2'], inplace=True)
compared_words_list_2.sort_values(by=['word1', 'word2'], inplace=True)

compared_words_list_1['easier_word']
compared_words_list_2['easier_word']

consistency_bool_list = list()
for k in range(len(compared_words_list_2)):
    word_inlist1 = compared_words_list_1.easier_word.tolist()[k]
    word_inlist2 = compared_words_list_2.easier_word.tolist()[k]
    if word_inlist1 != word_inlist2:
        consistency_bool_list.append(0)
    else:
        consistency_bool_list.append(1)

compared_words_list_1['consistency'] = consistency_bool_list
print(sum(consistency_bool_list)/len(consistency_bool_list))

# pick only consistent words

consistent_word_list = compared_words_list_1[compared_words_list_1['consistency']==1]

calc_accuracy(consistent_word_list, word_list)
calc_accuracy(compared_words_list, word_list)
