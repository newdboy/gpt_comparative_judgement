import pandas as pd
from ftxt_embedding import .

# import word list
test_source = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/평가 데이터/김한샘어휘_20220902.xlsx'
df = pd.read_excel(test_source)
df.head()
len(df)
len(df_dd)
df_dd = df.drop_duplicates(subset=['단어'])
# compare

inlist_comparison=list()
for word1, word2 in tqdm(list(product(*[,tbs_2021]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    fam_sim_2021.append(tmp)


df['단어'][df.duplicated(['단어'])].tolist()

for k in range(len(df)):
    if df['학년군'].iloc[k] == '1~2':
        if df['수준별등급'].iloc[k] == '상':
            df.iloc[k]['new_level'] = 2.0
        elif df['수준별등급'].iloc[k] == '중':
            df.iloc[k]['new_level'] = 1.5
        elif df['수준별등급'].iloc[k] == '하':
            df.iloc[k]['new_level'] = 1.0

    if df['학년군'].iloc[k] == '3~4':
        if df['수준별등급'].iloc[k] == '상':
            df.iloc[k]['new_level'] = 4.0
        elif df['수준별등급'].iloc[k] == '중':
            df.iloc[k]['new_level'] = 3.5
        elif df['수준별등급'].iloc[k] == '하':
            df.iloc[k]['new_level'] = 3.0

    if df['학년군'].iloc[k] == '5~6':
        if df['수준별등급'].iloc[k] == '상':
            df.iloc[k]['new_level'] = 6.0
        elif df['수준별등급'].iloc[k] == '중':
            df.iloc[k]['new_level'] = 5.5
        elif df['수준별등급'].iloc[k] == '하':
            df.iloc[k]['new_level'] = 5.0

